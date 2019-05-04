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
	"regexp"
	"strconv"
	"sync"
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
	detectionPadding      float64 = 1.5
	distance                      = 2.5
	videoFile                     = ""
	motionFile                    = ""
	notificationURL               = ""
	motionNotificationURL         = ""
	mbx                   int     = 42 // 120
	mby                   int     = 24 // 68
	magnitude             int     = 20 // 60
	totalMotion           int     = 4  // 10
	motionThrottle                = 1 * time.Minute
	normalizeEmbedding            = false
	outputFaces                   = false
	addr                          = ":9081"
	faceCacheSize                 = 100

	facesPathRegexp = regexp.MustCompile("(/((\\d+)|frame)(/(image|mimeType|time|width|height))?)?")
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

func Norm(e []float32) float32 {
	acc := float32(0.)

	for i := 0; i < len(e); i++ {
		acc += e[i] * e[i]
	}

	return float32(math.Sqrt(float64(acc)))
}

func scaleRectangle(r image.Rectangle, factor float64) image.Rectangle {
	dx := int(float64(r.Dx()) * (factor - 1.0) / 2)
	dy := int(float64(r.Dy()) * (factor - 1.0) / 2)

	return image.Rect(r.Min.X-dx, r.Min.Y-dy, r.Max.X+dx, r.Max.Y+dy)
}

type Face struct {
	Image    []byte    `json:"image"`
	MimeType string    `json:"mimeType"`
	Time     time.Time `json:"time"`
	Width    int       `json:"width"`
	Height   int       `json:"height"`
}

type FacesHandler struct {
	frames       [2]image.Image
	currentFrame int
	Faces        []Face
	CacheSize    int
	Quality      int
	locker       sync.Locker
}

func NewFacesHandler(maxSize int) *FacesHandler {
	return &FacesHandler{
		CacheSize: maxSize,
		locker:    new(sync.Mutex),
		Quality:   75,
	}
}

func (h *FacesHandler) Add(face image.Image) {
	h.locker.Lock()
	defer h.locker.Unlock()

	b := new(bytes.Buffer)
	if err := jpeg.Encode(b, face, &jpeg.Options{Quality: h.Quality}); err != nil {
		log.Printf("error encoding jpeg: %v", err)
	}
	h.Faces = append(h.Faces, Face{
		Image:    b.Bytes(),
		MimeType: "image/jpeg",
		Time:     time.Now(),
		Width:    face.Bounds().Dx(),
		Height:   face.Bounds().Dy(),
	})

	if len(h.Faces) > h.CacheSize {
		h.Faces = h.Faces[len(h.Faces)-h.CacheSize:]
	}
}

func (h *FacesHandler) Frame(frame image.Image) {
	h.locker.Lock()
	defer h.locker.Unlock()

	h.currentFrame = (h.currentFrame + 1) % len(h.frames)
	h.frames[h.currentFrame] = frame
}

func (h *FacesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := facesPathRegexp.FindStringSubmatch(r.URL.Path)

	log.Printf("path: %s, %#v", r.URL.Path, path)

	h.locker.Lock()
	defer h.locker.Unlock()

	var obj interface{} = h.Faces
	mimeType := ""
	var image []byte

	if path[3] != "" {
		if dex, err := strconv.ParseInt(path[3], 10, 32); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		} else if dex < 0 || int(dex) >= len(h.Faces) {
			http.Error(w, "invalid index", http.StatusNotFound)
			return
		} else if path[4] == "" {
			obj = h.Faces[dex]
		} else {
			switch path[4] {
			case "image":
				image = h.Faces[dex].Image
				mimeType = h.Faces[dex].MimeType
			case "mimeType":
				obj = h.Faces[dex].MimeType
			case "time":
				obj = h.Faces[dex].Time
			case "width":
				obj = h.Faces[dex].Width
			case "height":
				obj = h.Faces[dex].Height
			default:
				http.Error(w, "not found", http.StatusNotFound)
				return
			}
		}
	} else if path[2] == "frame" {
		if h.frames[h.currentFrame] == nil {
			http.Error(w, "no frames", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "image/jpeg")
		if err := jpeg.Encode(w, h.frames[h.currentFrame], &jpeg.Options{Quality: h.Quality}); err != nil {
			log.Printf("jpeg encoding error: %v", err)
		}
		return
	}

	if image != nil {
		// return the actual image
		w.Header().Set("Content-Type", mimeType)
		w.Write(image)
		return
	} else {
		if b, err := json.Marshal(obj); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		} else {
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write(b); err != nil {
				log.Printf("error writing to response: %v", err)
			}
		}
		return
	}
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
	flag.StringVar(&motionNotificationURL, "motionNotificationURL", motionNotificationURL, "url to notify of motion")
	flag.IntVar(&mbx, "mbx", mbx, "motion x vector size")
	flag.IntVar(&mby, "mby", mby, "motion y vector size")
	flag.IntVar(&magnitude, "magnitude", magnitude, "motion magnitude")
	flag.IntVar(&totalMotion, "totalMotion", totalMotion, "total high magnitude motion counts to trigger motion detection")
	flag.Float64Var(&detectionPadding, "padding", detectionPadding, "padding of face detection rectangles")
	flag.DurationVar(&motionThrottle, "motionThrottle", motionThrottle, "duration to throttle motion by")
	flag.BoolVar(&normalizeEmbedding, "normalizeEmbedding", normalizeEmbedding, "normalize embedding prior to distance calculation")
	flag.BoolVar(&outputFaces, "outputFaces", outputFaces, "output faces to directory")
	flag.StringVar(&addr, "addr", addr, "service address")
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

func (m MotionProcessor) Notify(magnitude float64) {
	client := &http.Client{
		Timeout: 1000 * time.Millisecond,
	}

	msg := "on"

	if bb, err := json.Marshal(msg); err != nil {
		log.Fatal(err)
	} else if req, err := http.NewRequest("POST", m.NotificationURL, bytes.NewReader(bb)); err != nil {
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
		}

		c := 0
		for _, v := range vect {
			magU := int(v.X)*int(v.X) + int(v.Y)*int(v.Y)
			if magU > mag2 {
				c++
			}
		}

		// log.Printf("total motion vectors above magnitude: %d", c)

		if c > m.Total && time.Now().Sub(last) > m.Throttle {
			last = time.Now()

			go m.Notify(float64(c))
		}

	}
	log.Print("finishing motion processor")
}

func main() {
	flag.Parse()

	facesHandler := NewFacesHandler(faceCacheSize)
	server := &http.Server{
		Addr:    addr,
		Handler: facesHandler,
	}
	go func() {
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()
	defer server.Close()

	people := People{}
	mot := MotionProcessor{
		MBX:             mbx,
		MBY:             mby,
		Magnitude:       magnitude,
		Total:           totalMotion,
		Throttle:        motionThrottle,
		NotificationURL: motionNotificationURL,
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

	go func() {
		log.Printf("opening motion file: '%s'", motionFile)
		var m *os.File
		if motionFile != "" {
			var err error
			if m, err = os.OpenFile(motionFile, os.O_RDONLY, 0600); err != nil {
				log.Fatal(err)
			} else {
				defer m.Close()
				mot.ProcessMotion(m)
			}
		}
	}()

	r := os.Stdin
	if videoFile != "" {
		var err error
		if r, err = os.OpenFile(videoFile, os.O_RDONLY, 0600); err != nil {
			log.Fatal(err)
		}
	}
	reader := detector.RGB24Reader{
		Reader: r,
		Rect:   image.Rect(0, 0, imageWidth, imageHeight),
	}

	j := 0

	if outputFaces {
		os.Mkdir("faces", 0770)
	}

	for {
		var rgb *detector.RGB24
		var err error

		if rgb, err = reader.ReadRGB24(); err != nil {
			log.Print(err)
			break
		}

		facesHandler.Frame(rgb)
		detections := det.InferRGB(rgb)

		// log.Printf("found: %d", len(detections))

		for _, d := range detections {
			// log.Printf("%d: confidence: %f, (%d %d) - (%d %d)", i, d.Confidence, d.Rect.Min.X, d.Rect.Min.Y, d.Rect.Max.X, d.Rect.Max.Y)

			// padd the rectangle to get more of the face
			r := scaleRectangle(d.Rect, detectionPadding)

			if !r.In(reader.Rect) {
				// out of bounds
				continue
			}

			face := rgb.SubImage(r)
			facesHandler.Add(face)

			classification := classer.InferRGB24(face.(*detector.RGB24))

			// go mot.Notify(0)

			if normalizeEmbedding {
				n := Norm(classification.Embedding)
				for i, x := range classification.Embedding {
					classification.Embedding[i] = x / n
				}
			}

			if outputFaces {
				if f, err := os.OpenFile(fmt.Sprintf("faces/face%05d.jpg", j), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0660); err != nil {
					log.Fatal(err)
				} else if err := jpeg.Encode(f, face, nil); err != nil {
					log.Fatal(err)
				} else {
					f.Close()
				}

				if f, err := os.OpenFile(fmt.Sprintf("faces/face%005d.json", j), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0660); err != nil {
					log.Fatal(err)
				} else if b, err := json.Marshal(classification); err != nil {
					log.Fatal(err)
				} else if _, err := f.Write(b); err != nil {
					log.Fatal(err)
				} else {
					f.Close()
				}
			}

			for _, p := range people {
				if d := Dist(classification.Embedding, p.Embedding); d < float32(distance) {
					// log.Printf("match: %s", p.Name)

					go notify(p.Name, face)
				}
			}
		}

		j++
		j = j % 10000
	}
}
