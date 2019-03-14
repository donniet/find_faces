package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"time"

	detector "github.com/donniet/detector"
)

var (
	detectorDescription           = ""
	detectorWeights               = ""
	classifierDescription         = ""
	classifierWeights             = ""
	deviceName                    = ""
	peopleFile                    = ""
	imageWidth                    = 672
	imageHeight                   = 384
	numChannels                   = 3
	detectionPadding      float32 = 1.275
	distance                      = 2.5
	videoFile                     = ""
	motionFile                    = ""
	notificationURL               = ""
	mbx                   int     = 120
	mby                   int     = 68
	magnitude             int     = 60
	totalMotion           int     = 10
)

type People []Person

type Person struct {
	Name      string    `json:"name"`
	Embedding []float32 `json:"embedding"`
}

func Dist(e []float32, r []float32) float32 {
	acc := float32(0.)

	if len(e) != len(r) {
		panic(fmt.Errorf("cannot calculate distance on embeddings of different sizes"))
	}

	for i := 0; i < len(e); i++ {
		d := e[i] - r[i]
		acc += d * d
	}

	return float32(math.Sqrt(float64(acc)))
}

func scaleRectangle(r image.Rectangle, factor float32) image.Rectangle {
	dx := int(float32(r.Dx()) * (factor - 1.0) / 2)
	dy := int(float32(r.Dy()) * (factor - 1.0) / 2)

	return image.Rect(r.Min.X-dx, r.Min.Y-dy, r.Max.X+dx, r.Max.Y+dy)
}

func init() {
	flag.StringVar(&detectorDescription, "detectorDescription", detectorDescription, "detector description file")
	flag.StringVar(&detectorWeights, "detectorWeights", detectorWeights, "detector weights file")
	flag.StringVar(&classifierDescription, "classifierDescription", classifierDescription, "classifier description file")
	flag.StringVar(&classifierWeights, "classifierWeights", classifierWeights, "classifier weights file")
	flag.StringVar(&peopleFile, "peopleFile", peopleFile, "people file")
	flag.StringVar(&deviceName, "device", deviceName, "device name")
	flag.Float64Var(&distance, "distance", distance, "distance before identifying match")
	flag.StringVar(&videoFile, "video", videoFile, "video file (leave blank for stdin)")
	flag.StringVar(&motionFile, "motion", motionFile, "motion pipe (leave blank to ignore motion)")
	flag.IntVar(&imageWidth, "width", imageWidth, "image width")
	flag.IntVar(&imageHeight, "height", imageHeight, "image height")
	flag.StringVar(&notificationURL, "notificationURL", notificationURL, "url to notify of found faces")
	flag.IntVar(&mbx, "mbx", mbx, "motion x vector size")
	flag.IntVar(&mby, "mby", mby, "motion y vector size")
	flag.IntVar(&magnitude, "magnitude", magnitude, "motion magnitude")
	flag.IntVar(&totalMotion, "totalMotion", totalMotion, "total high magnitude motion counts to trigger motion detection")
	flag.Parse()
}

func notify(name string, face image.Image) {
	if notificationURL == "" {
		return
	}

	b := &bytes.Buffer{}
	client := &http.Client{
		Timeout: 1000 * time.Millisecond,
	}

	if err := jpeg.Encode(b, face, nil); err != nil {
		panic(err)
	}

	imageEncoded := base64.StdEncoding.EncodeToString(b.Bytes())

	msg := map[string]interface{}{
		"name":  name,
		"image": "image/jpeg;base64," + imageEncoded,
	}

	if bb, err := json.Marshal(msg); err != nil {
		panic(err)
	} else if req, err := http.NewRequest("PUT", notificationURL, bytes.NewReader(bb)); err != nil {
		panic(err)
	} else if res, err := client.Do(req); err != nil {
		log.Print(err)
	} else if res.StatusCode >= 300 || res.StatusCode < 200 {
		log.Printf("unknown status code: %d %s", res.StatusCode, res.Status)
	}
}

type MotionProcessor struct {
	MBX             int
	MBY             int
	Magnitude       int
	Total           int
	Throttle        time.Duration
	NotificationURL string
}

type motionVector struct {
	X   int8
	Y   int8
	Sad int16
}

func (m MotionProcessor) notify(magnitude float64) {
	client := &http.Client{
		Timeout: 1000 * time.Millisecond,
	}

	msg := map[string]interface{}{
		"magnitude": magnitude,
	}

	if bb, err := json.Marshal(msg); err != nil {
		log.Fatal(err)
	} else if req, err := http.NewRequest("PUT", m.NotificationURL, bytes.NewReader(bb)); err != nil {
		log.Fatal(err)
	} else if res, err := client.Do(req); err != nil {
		log.Print(err)
	} else if res.StatusCode < 200 || res.StatusCode >= 300 {
		log.Print(fmt.Errorf("error code from motion processor notification: %d %s", res.StatusCode, res.Status))
	}
}

func (m MotionProcessor) ProcessMotion(r io.Reader) {
	len := (m.MBX + 1) * m.MBY
	vect := make([]motionVector, len)

	mag2 := m.Magnitude * m.Magnitude

	last := time.Time{}

	for {
		if err := binary.Read(r, binary.LittleEndian, vect); err != nil {
			log.Print(err)
			break
		} else if time.Now().Sub(last) < m.Throttle {
			continue
		}
		last = time.Now()

		c := 0
		for _, v := range vect {
			magU := int(v.X)*int(v.X) + int(v.Y)*int(v.Y)
			if magU > mag2 {
				c++
			}
		}

		// log.Printf("total motion vectors above magnitude: %d", c)

		if c > m.Total {
			go m.notify(float64(c))
		}

	}
	log.Print("finishing motion processor")
}

func main() {
	people := People{}
	mot := MotionProcessor{
		MBX:       mbx,
		MBY:       mby,
		Magnitude: magnitude,
		Total:     totalMotion,
	}

	if f, err := os.Open(peopleFile); err != nil {
		log.Printf("person file not found '%s'", peopleFile)
	} else if b, err := ioutil.ReadAll(f); err != nil {
		log.Fatal(err)
	} else if err := json.Unmarshal(b, &people); err != nil {
		log.Fatal(err)
	}

	det := detector.NewDetector(detectorDescription, detectorWeights, deviceName)
	defer det.Close()

	classer := detector.NewClassifier(classifierDescription, classifierWeights, deviceName)
	defer classer.Close()

	r := os.Stdin
	if videoFile != "" {
		var err error
		if r, err = os.OpenFile(videoFile, os.O_RDONLY, 0600); err != nil {
			log.Fatal(err)
		}
	}

	var m *os.File
	if motionFile != "" {
		var err error
		if m, err = os.OpenFile(motionFile, os.O_RDONLY, 0600); err != nil {
			log.Fatal(err)
		} else {
			defer m.Close()
			go mot.ProcessMotion(m)
		}
	}

	reader := detector.RGB24Reader{
		Reader: r,
		Rect:   image.Rect(0, 0, imageWidth, imageHeight),
	}

	for {
		var rgb *detector.RGB24
		var err error

		if rgb, err = reader.ReadRGB24(); err != nil {
			log.Print(err)
			break
		}

		detections := det.InferRGB(rgb)

		log.Printf("found: %d", len(detections))

		for _, d := range detections {
			// log.Printf("%d: confidence: %f, (%d %d) - (%d %d)", i, d.Confidence, d.Rect.Min.X, d.Rect.Min.Y, d.Rect.Max.X, d.Rect.Max.Y)

			// padd the rectangle to get more of the face
			r := scaleRectangle(d.Rect, detectionPadding)

			if !r.In(reader.Rect) {
				// out of bounds
				continue
			}

			face := rgb.SubImage(r)

			classification := classer.InferRGB24(face.(*detector.RGB24))

			for _, p := range people {
				if d := Dist(classification.Embedding, p.Embedding); d < float32(distance) {
					log.Printf("match: %s", p.Name)

					go notify(p.Name, face)
				}
			}
		}
	}
}
