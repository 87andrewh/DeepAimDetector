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

// Defines amount of frames to collect around kills
const samplesPerSecond = 32
const secondsBeforeKill = 2
const secondsAfterKill = 32.0 / 32
const secondsPerKill = secondsBeforeKill + secondsAfterKill
const samplesPerKill = int(secondsPerKill * samplesPerSecond)

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

// KillTime stores the frames around a kill.
type KillTime struct {
	killer     int
	victim     int
	startFrame int
	killFrame  int
	endFrame   int
}

// FireFrameKey is a key to a dictionary that marks
// if a shooter shoot at a given frame
type FireFrameKey struct {
	shooter int
	frame   int
}

// KillData stores the features of a single sample fed into the model.
type KillData struct {
	// Whether the killer used an aimbot during the kill
	killerAimbot bool

	// One-hot encoding of killing gun
	weaponAK47 bool
	weaponM4A4 bool
	weaponAWP  bool

	// Viewangle deltas
	killerDeltaYaw   [samplesPerKill]float32
	killerDeltaPitch [samplesPerKill]float32

	// Angles between the killer's crosshair and the victim
	crosshairToVictimYaw   [samplesPerKill]float32
	crosshairToVictimPitch [samplesPerKill]float32

	victimDistance  [samplesPerKill]float32
	killerCrouching [samplesPerKill]bool
	victimCrouching [samplesPerKill]bool
	killerFiring    [samplesPerKill]bool
	victimHealth    [samplesPerKill]int
}

// Marks guns that the model will be trained on
// TODO: Test model on different sets of guns.
var validGuns = map[string]bool{
	"AK-47": true,
	"M4A4":  true,
	"AWP":   true,
	//"M4A1": true,
	//"AUG":    true,
	//"SG 553": true,
}

// Stores data to be fed into model
var modelData = []KillData{}

func main() {
	dir := "./demos/labeled/"
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		log.Fatal(err)
	}
	for _, f := range files {
		fmt.Println(f.Name())
		parseDemo(dir + f.Name())
	}
	csvExport()
}

func parseDemo(path string) {
	// Times when a player is killed by a valid gun
	var killTimes = []KillTime{}
	// Marks if a player is firing at a given frame.
	var fireFrames = map[FireFrameKey]bool{}
	// Marks frames surrounding kills that should be gathered
	// for easier processing into model inputs
	var isMarked = map[int]bool{}
	// Stores the PlayerData for each player for each marked framed
	var markedFrameData = map[int]map[int]PlayerData{}
	// Marks if the demo was generated with an aimbot
	aimbot := strings.Contains(path, "_aimbot_")

	f, err := os.Open(path)
	defer f.Close()
	checkError(err)
	p := dem.NewParser(f)

	// Calculate the demo framerate with some hacks
	tick := -1
	for !(10 < tick && tick < 100) {
		_, err = p.ParseNextFrame()
		checkError(err)
		tick = p.GameState().IngameTick()
	}
	_, err = p.ParseNextFrame()
	checkError(err)
	nextTick := p.GameState().IngameTick()
	FrameRate := int(p.TickRate()) / (nextTick - tick)

	var framesBeforeKill int
	var framesAfterKill int
	if FrameRate == 32 {
		framesBeforeKill = secondsBeforeKill * 32
		framesAfterKill = secondsAfterKill * 32
	} else if FrameRate == 64 {
		framesBeforeKill = secondsBeforeKill * 64
		framesAfterKill = secondsAfterKill * 64
	} else if FrameRate == 128 {
		framesBeforeKill = secondsBeforeKill * 128
		framesAfterKill = secondsAfterKill * 128
	} else {
		println("Invalid frame rate: ", FrameRate)
		return
	}
	framesPerKill := framesBeforeKill + framesAfterKill
	framesPerSample := int(framesPerKill / samplesPerKill)
	println("Frames per sample ", framesPerSample)

	// First pass.

	// Get frame times of kills with valid guns,
	// and mark surrounding frames for retrieval.
	killCount := 0
	p.RegisterEventHandler(func(e events.Kill) {
		if !validGuns[e.Weapon.String()] {
			return
		}
		if e.Killer.SteamID64 == 0 { // Ignore bots
			return
		}

		killCount++
		killFrame := p.CurrentFrame()
		start := killFrame - framesBeforeKill
		end := killFrame + framesAfterKill
		for frame := start; frame < end; frame++ {
			isMarked[frame] = true
		}
		isMarked[start-framesPerSample] = true // For first sample delta angles
		newKillTime := KillTime{
			e.Killer.UserID, e.Victim.UserID, start, killFrame, end}
		killTimes = append(killTimes, newKillTime)
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
	fmt.Printf("Kills with valid guns: %d\n", killCount)
	checkError(err)
	f.Close()

	// Second pass.

	// Extract player data from marked frames
	f, err = os.Open(path)
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
			if player.ActiveWeapon() == nil {
				continue
			}
			players[player.UserID] = extractPlayerData(frame, player, fireFrames)
		}
		markedFrameData[frame] = players
	}
	checkError(err)

	// Extract each kill's KillData, and add it to modelData
	for _, kill := range killTimes {
		weapon := markedFrameData[kill.killFrame][kill.killer].weapon
		killData := KillData{
			killerAimbot: aimbot,
			weaponAK47:   weapon == "AK-47",
			weaponM4A4:   weapon == "M4A4",
			weaponAWP:    weapon == "AWP",
		}

		prevFrame := kill.startFrame - framesPerSample
		prevKillerYaw := markedFrameData[prevFrame][kill.killer].yaw
		prevKillerPitch := markedFrameData[prevFrame][kill.killer].pitch

		for sample := 0; sample < samplesPerKill; sample++ {
			frame := framesPerSample*sample + kill.startFrame
			killer := markedFrameData[frame][kill.killer]
			victim := markedFrameData[frame][kill.victim]

			killerYaw := killer.yaw
			killerPitch := killer.pitch
			killData.killerDeltaYaw[sample] = normalizeAngle(
				killerYaw - prevKillerYaw)
			killData.killerDeltaPitch[sample] = killerPitch - prevKillerPitch
			prevKillerYaw = killerYaw
			prevKillerPitch = killerPitch

			killerToVictim := victim.position.Sub(killer.position)
			dX := killerToVictim.X
			dY := killerToVictim.Y
			dZ := killerToVictim.Z
			killerToVictimYaw := 180 / math.Pi * float32(math.Atan2(dY, dX))
			killerToVictimPitch := 180 / math.Pi * float32(math.Atan2(
				math.Sqrt(dX*dX+dY*dY),
				dZ))
			// Smallest angle between killerToVictimYaw and killerYaw
			killData.crosshairToVictimYaw[sample] =
				normalizeAngle(killerToVictimYaw - killerYaw)
			killData.crosshairToVictimPitch[sample] =
				killerToVictimPitch - killerPitch

			killData.victimDistance[sample] = float32(killerToVictim.Norm())
			killData.killerCrouching[sample] = killer.crouching
			killData.victimCrouching[sample] = victim.crouching
			killData.killerFiring[sample] = killer.firing
			killData.victimHealth[sample] = victim.health
		}
		modelData = append(modelData, killData)
	}

	//kill := modelData[6]
	//for i := 0; i < samplesPerKill; i++ {
	//	fmt.Printf(
	//		"(%.3f %.3f) ",
	//		kill.crosshairToVictimPitch[i], kill.crosshairToVictimYaw[i])
	//	fmt.Printf(
	//		"(%.3f %.3f) ",
	//		kill.killerDeltaPitch[i], kill.killerDeltaYaw[i])
	//	fmt.Printf(" %d ", kill.victimHealth[i])
	//	fmt.Printf(" %t ", kill.killerFiring[i])
	//	//fmt.Printf("%t ", kill.killerCrouching[i])
	//	//fmt.Printf("%d\n", kill.victimHealth[i])S
	//	println(i)
	//}
}

func extractPlayerData(
	frame int,
	player *common.Player,
	fireFrames map[FireFrameKey]bool) PlayerData {

	fixedPitch := float32(math.Mod(
		float64(player.ViewDirectionY())+90,
		180))
	return PlayerData{
		player.ActiveWeapon().String(),
		player.LastAlivePosition,
		player.ViewDirectionX(),
		fixedPitch,
		player.IsDucking(),
		fireFrames[FireFrameKey{player.UserID, frame}],
		player.Health()}
}

func csvExport() error {
	file, err := os.OpenFile("./aim.csv", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)

	for _, killData := range modelData {
		err := writer.Write(killToString(killData))
		if err != nil {
			return err
		}
	}

	writer.Flush()
	file.Close()
	return nil
}

func killToString(data KillData) []string {
	var out []string

	var aimbot int
	if data.killerAimbot {
		aimbot = 1
	} else {
		aimbot = 0
	}
	out = append(out, fmt.Sprintf("%d", aimbot))

	for i := 0; i < samplesPerKill; i++ {
		out = append(out, fmt.Sprintf("%.3f", data.killerDeltaYaw[i]))
		out = append(out, fmt.Sprintf("%.3f", data.killerDeltaPitch[i]))
		out = append(out, fmt.Sprintf("%.3f", data.crosshairToVictimYaw[i]))
		out = append(out, fmt.Sprintf("%.3f", data.crosshairToVictimPitch[i]))
		out = append(out, fmt.Sprintf("%.0f", data.victimDistance[i]))
		var firing int
		if data.killerFiring[i] {
			firing = 1
		} else {
			firing = 0
		}
		out = append(out, fmt.Sprintf("%d", firing))
		out = append(out, fmt.Sprintf("%d", int(data.victimHealth[i])))
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
