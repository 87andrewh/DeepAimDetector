package main

import (
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"

	strings "strings"

	"github.com/golang/geo/r3"
	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	common "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
	events "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

// File paths
var dest = "../model/other.csv"
var source = "./demos/other/"

// Defines amount of frames to collect around attacks
const samplesPerSecond = 32
const secondsBeforeAttack = 1
const secondsAfterAttack = 1
const secondsPerAttack = secondsBeforeAttack + secondsAfterAttack
const samplesPerAttack = int(samplesPerSecond * secondsPerAttack)

// PlayerData stores all data of a player in a single frame.
type PlayerData struct {
	weapon    string
	position  r3.Vector
	yaw       float32
	pitch     float32
	crouching bool
	firing    bool
	health    int
}

// AttackTime marks when an attacker shot a victim
type AttackTime struct {
	attacker    int
	victim      int
	startFrame  int
	attackFrame int
	endFrame    int
}

// FireFrameKey is a key to a dictionary that marks
// if a shooter shoot at a given frame
type FireFrameKey struct {
	shooter int
	frame   int
}

// AttackData stores the features of a single sample fed into the model.
type AttackData struct {
	// Whether the attacker used an aimbot during the attack
	attackerAimbot bool

	// One-hot encoding of attacking gun
	weaponAK47 bool
	weaponM4A4 bool
	weaponAWP  bool

	// Viewangle deltas
	attackerDeltaYaw   [samplesPerAttack]float32
	attackerDeltaPitch [samplesPerAttack]float32

	// Angles between the attacker's crosshair and the victim
	crosshairToVictimYaw   [samplesPerAttack]float32
	crosshairToVictimPitch [samplesPerAttack]float32

	victimDistance    [samplesPerAttack]float32
	attackerCrouching [samplesPerAttack]bool
	victimCrouching   [samplesPerAttack]bool
	attackerFiring    [samplesPerAttack]bool

	attackerHealth [samplesPerAttack]int
	victimHealth   [samplesPerAttack]int

	attackerViewVector     [samplesPerAttack]r3.Vector
	attackerToVictimVector [samplesPerAttack]r3.Vector

	//attackerX [samplesPerAttack]float32
	//victimX   [samplesPerAttack]float32
}

// Marks guns that the model will be trained on
// TODO: Test model on different sets of guns.
var validGuns = map[string]bool{
	"AK-47": true,
	//"M4A4":  true,
	//"AWP":   true,
	//"M4A1": true,
	//"AUG":    true,
	//"SG 553": true,
}

// Stores data to be fed into model
var modelData = []AttackData{}

func main() {
	//source = "E:/demos/shadowkeeper/"

	files, err := ioutil.ReadDir(source)
	if err != nil {
		log.Fatal(err)
	}
	for _, f := range files {
		fmt.Println(f.Name())
		parseDemo(source, f.Name())
	}
	csvExport()
}

func parseDemo(source string, name string) {
	// Times when a player is attacked by a valid gun
	var attackTimes = []AttackTime{}
	// Marks if a player is firing at a given frame.
	var fireFrames = map[FireFrameKey]bool{}
	// Marks frames surrounding attacks that should be gathered
	// for easier processing into model inputs
	var isMarked = map[int]bool{}
	// Stores the PlayerData for each player for each marked framed
	var markedFrameData = map[int]map[int]PlayerData{}
	// Marks if the demo was generated with an aimbot
	aimbot := strings.Contains(name, "_aimbot_")

	f, err := os.Open(source + name)
	defer f.Close()
	checkError(err)
	p := dem.NewParser(f)

	h, err := p.ParseHeader()
	FrameRate := h.FrameRate()

	// Calculate the demo framerate with some hacks
	tick := -1
	for !(2900 < tick && tick < 3000) {
		_, err = p.ParseNextFrame()
		tick = p.GameState().IngameTick()
	}
	checkError(err)
	iters := 10
	for i := 0; i < iters; i++ {
		_, err = p.ParseNextFrame()
		checkError(err)
	}
	nextTick := p.GameState().IngameTick()

	TicksPerFrame := float64(nextTick-tick) / float64(iters)
	FrameRate2 := p.TickRate() / TicksPerFrame

	if FrameRate == 0 {
		FrameRate = FrameRate2
	}

	var framesBeforeAttack int
	var framesAfterAttack int
	if (math.Abs(FrameRate-32.0) < 1) && (FrameRate2 == 32) {
		framesBeforeAttack = secondsBeforeAttack * 32
		framesAfterAttack = secondsAfterAttack * 32
	} else if (math.Abs(FrameRate-64.0) < 4) && (FrameRate2 == 64) {
		framesBeforeAttack = secondsBeforeAttack * 64
		framesAfterAttack = secondsAfterAttack * 64
	} else if (math.Abs(FrameRate-128) < 4) && (FrameRate2 == 128) {
		framesBeforeAttack = secondsBeforeAttack * 128
		framesAfterAttack = secondsAfterAttack * 128
	} else {
		println("Invalid frame rate: ", FrameRate, FrameRate2)
		return
	}

	framesPerAttack := framesBeforeAttack + framesAfterAttack
	framesPerSample := int(framesPerAttack / samplesPerAttack)
	println("Frames per sample ", framesPerSample)

	// First pass.

	// Get frame times of attacks with valid guns,
	// and mark surrounding frames for retrieval.
	attackCount := 0
	p.RegisterEventHandler(func(e events.PlayerHurt) {
		if !validGuns[e.Weapon.String()] {
			return
		}
		if e.Attacker.SteamID64 == 0 { // Ignore bots
			return
		}

		attackCount++
		attackFrame := p.CurrentFrame()
		start := attackFrame - framesBeforeAttack
		end := attackFrame + framesAfterAttack
		for frame := start; frame < end; frame++ {
			isMarked[frame] = true
		}
		isMarked[start-framesPerSample] = true // For first sample delta angles
		new := AttackTime{
			e.Attacker.UserID, e.Player.UserID, start, attackFrame, end}
		attackTimes = append(attackTimes, new)
	})

	// Track frames where a player fires a weapon
	p.RegisterEventHandler(func(e events.WeaponFire) {
		frame := p.CurrentFrame()
		// Include previous frames so that shot is not lost after sampling
		for i := 0; i < framesPerSample; i++ {
			fireFrames[FireFrameKey{e.Shooter.UserID, frame - i}] = true
		}
	})
	err = p.ParseToEnd()
	fmt.Printf("Valid attacks: %d\n", attackCount)

	// Second pass.

	// Extract player data from marked frames
	f, err = os.Open(source + name)
	p = dem.NewParser(f)
	for ok := true; ok; ok, err = p.ParseNextFrame() {
		checkError(err)
		frame := p.CurrentFrame()

		if !isMarked[frame] {
			continue
		}

		var players = map[int]PlayerData{}
		gs := p.GameState()
		for _, player := range gs.Participants().Playing() {
			players[player.UserID] = extractPlayerData(frame, player, fireFrames)
		}
		markedFrameData[frame] = players
	}

	// Extract each attack's AttackData, and add it to modelData
	for _, attack := range attackTimes {
		weapon := markedFrameData[attack.attackFrame][attack.attacker].weapon
		attackData := AttackData{
			attackerAimbot: aimbot,
			weaponAK47:     weapon == "AK-47",
			weaponM4A4:     weapon == "M4A4",
			weaponAWP:      weapon == "AWP",
		}

		prevFrame := attack.startFrame - framesPerSample
		prevAttackerYaw := markedFrameData[prevFrame][attack.attacker].yaw
		prevAttackerPitch := markedFrameData[prevFrame][attack.attacker].pitch

		for sample := 0; sample < samplesPerAttack; sample++ {
			frame := framesPerSample*sample + attack.startFrame
			attacker := markedFrameData[frame][attack.attacker]
			victim := markedFrameData[frame][attack.victim]

			attackerYaw := attacker.yaw
			attackerPitch := attacker.pitch
			attackData.attackerDeltaYaw[sample] = normalizeAngle(
				attackerYaw - prevAttackerYaw)
			attackData.attackerDeltaPitch[sample] = attackerPitch - prevAttackerPitch
			prevAttackerYaw = attackerYaw
			prevAttackerPitch = attackerPitch

			attackerToVictim := victim.position.Sub(attacker.position)
			attackData.attackerToVictimVector[sample] = attackerToVictim.Normalize()

			dX := attackerToVictim.X
			dY := attackerToVictim.Y
			dZ := attackerToVictim.Z
			attackerToVictimYaw := 180 / math.Pi * float32(math.Atan2(dY, dX))
			attackerToVictimPitch := 180 / math.Pi * float32(math.Atan2(
				math.Sqrt(dX*dX+dY*dY),
				dZ))

			// Smallest angle between attackerToVictimYaw and attackerYaw
			attackData.crosshairToVictimYaw[sample] =
				normalizeAngle(attackerToVictimYaw - attackerYaw)
			attackData.crosshairToVictimPitch[sample] =
				attackerToVictimPitch - attackerPitch

			attackData.victimDistance[sample] = float32(attackerToVictim.Norm())

			attackData.attackerCrouching[sample] = attacker.crouching
			attackData.victimCrouching[sample] = victim.crouching
			attackData.attackerFiring[sample] = attacker.firing

			attackData.attackerHealth[sample] = attacker.health
			attackData.victimHealth[sample] = victim.health

			attackerYaw64 := float64(math.Pi / 180 * attackerYaw)
			attackerPitch64 := float64(math.Pi / 180 * attackerPitch)
			attackData.attackerViewVector[sample] = r3.Vector{
				math.Cos(attackerYaw64) * math.Sin(attackerPitch64),
				math.Sin(attackerYaw64) * math.Sin(attackerPitch64),
				math.Cos(attackerPitch64)}

			//attackData.attackerX[sample] = float32(attacker.position.X)
			//attackData.victimX[sample] = float32(victim.position.X)
		}
		// A player teleported. Throw away the data.
		modelData = append(modelData, attackData)
	}
	f.Close()
}

func extractPlayerData(
	frame int,
	player *common.Player,
	fireFrames map[FireFrameKey]bool) PlayerData {

	fixedPitch := float32(math.Mod(
		float64(player.ViewDirectionY())+90,
		180))

	weapon := ""
	if player.ActiveWeapon() != nil {
		weapon = player.ActiveWeapon().String()
	}

	return PlayerData{
		weapon,
		player.LastAlivePosition,
		player.ViewDirectionX(),
		fixedPitch,
		player.IsDucking(),
		fireFrames[FireFrameKey{player.UserID, frame}],
		player.Health()}
}

func csvExport() error {
	file, err := os.OpenFile(dest, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)

	for _, attackData := range modelData {
		err := writer.Write(attackToString(attackData))
		if err != nil {
			return err
		}
	}

	writer.Flush()
	file.Close()
	return nil
}

func attackToString(data AttackData) []string {
	var out []string

	var aimbot int
	if data.attackerAimbot {
		aimbot = 1
	} else {
		aimbot = 0
	}
	out = append(out, fmt.Sprintf("%d", aimbot))

	for i := 0; i < samplesPerAttack; i++ {
		//out = append(out, fmt.Sprintf("%.3f", data.attackerViewVector[i].X))
		//out = append(out, fmt.Sprintf("%.3f", data.attackerViewVector[i].Y))
		//out = append(out, fmt.Sprintf("%.3f", data.attackerViewVector[i].Z))
		//out = append(out, fmt.Sprintf("%.3f", data.attackerToVictimVector[i].X))
		//out = append(out, fmt.Sprintf("%.3f", data.attackerToVictimVector[i].Y))
		//out = append(out, fmt.Sprintf("%.3f", data.attackerToVictimVector[i].Z))
		//out = append(out, fmt.Sprintf("%.0f", data.victimDistance[i]))
		out = append(out, fmt.Sprintf("%.3f", data.attackerDeltaYaw[i]))
		out = append(out, fmt.Sprintf("%.3f", data.attackerDeltaPitch[i]))
		out = append(out, fmt.Sprintf("%.3f", data.crosshairToVictimYaw[i]))
		out = append(out, fmt.Sprintf("%.3f", data.crosshairToVictimPitch[i]))

		if data.attackerFiring[i] {
			out = append(out, fmt.Sprintf("%d", 1))
		} else {
			out = append(out, fmt.Sprintf("%d", 0))
		}

		if data.attackerHealth[i] == 0 {
			out = append(out, fmt.Sprintf("%d", 1))
		} else {
			out = append(out, fmt.Sprintf("%d", 0))
		}

		if data.victimHealth[i] == 0 {
			out = append(out, fmt.Sprintf("%d", 1))
		} else {
			out = append(out, fmt.Sprintf("%d", 0))
		}
	}
	return out
}

// Returns a mod b, keeping the sign of b
func divisorSignMod(a float64, b float64) float64 {
	return math.Mod(math.Mod(a, b)+b, b)
}

// Normalize an angle to be between -180 and 180
func normalizeAngle(a float32) float32 {
	return float32(-180 + divisorSignMod(float64(a)+180, 360))
}

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}
