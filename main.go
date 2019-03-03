package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync"

	"github.com/faiface/pixel"
	"github.com/faiface/pixel/pixelgl"
	"golang.org/x/image/colornames"
)

const (
	RowsPerGoroutine = 5
	WIDTH            = 300
	HEIGHT           = 300
	AntiAliasSample  = 100
	filename         = "image.ppm"
)

type HitRecord struct {
	t        float64
	p        *Vec
	normal   *Vec
	material Material
}

type Material interface {
	Scatter(ray *Ray, hitRecord *HitRecord) (scatter *Ray, attenuation *Vec)
}

func Reflect(vec *Vec, n *Vec) *Vec {
	return vec.Sub(n.MulConst(vec.Dot(n)).MulConst(2))
}

func Schlick(cosine float64, refractiveIndex float64) float64 {
	r0 := (1 - refractiveIndex) / (1 + refractiveIndex)
	r0 = r0 * r0
	return r0 + (1-r0)*math.Pow((1-cosine), 5.0)
}

func Refract(vec *Vec, n *Vec, niOverNt float64) *Vec {
	uv := vec.Unit()
	dt := uv.Dot(vec)
	discriminant := 1 - niOverNt*niOverNt*(1-dt*dt)
	if discriminant > 0 {
		return uv.Sub(n.MulConst(dt)).MulConst(niOverNt).Sub(n.MulConst(math.Sqrt(discriminant)))
	}
	return nil
}

type Metal struct {
	albedo *Vec
	fuzz   float64
}

func NewMetal(albedo *Vec, fuzz float64) *Metal {
	if fuzz > 1 {
		fuzz = 1
	}
	return &Metal{
		albedo: albedo,
		fuzz:   fuzz,
	}
}

func (metal *Metal) Scatter(ray *Ray, hitRecord *HitRecord) (scatter *Ray, attenuation *Vec) {
	reflected := Reflect(ray.Direction().Unit(), hitRecord.normal)
	scattered := NewRay(hitRecord.p, reflected.Add(RandomInUnitSphere().MulConst(metal.fuzz)))
	if scattered.Direction().Dot(hitRecord.normal) <= 0 {
		putRay(scattered)
		return nil, nil
	}
	return scattered, metal.albedo
}

type Dielectric struct {
	refractiveIndex float64
}

func NewDielectric(refractiveIndex float64) *Dielectric {
	return &Dielectric{
		refractiveIndex: refractiveIndex,
	}
}

func (dielectric *Dielectric) Scatter(ray *Ray, hitRecord *HitRecord) (scatter *Ray, attenuation *Vec) {
	var outwardNormal *Vec
	var reflectProb float64
	var cosine float64
	niOverNt := 0.0
	reflected := Reflect(ray.Direction(), hitRecord.normal)
	attenuation = NewVec(1, 1, 1)

	if ray.Direction().Dot(hitRecord.normal) > 0 {
		outwardNormal = hitRecord.normal.Neg()
		niOverNt = dielectric.refractiveIndex
		cosine = dielectric.refractiveIndex * ray.Direction().Dot(hitRecord.normal) / ray.Direction().Len()
	} else {
		outwardNormal = hitRecord.normal
		niOverNt = 1.0 / dielectric.refractiveIndex
		cosine = -ray.Direction().Dot(hitRecord.normal) / ray.Direction().Len()
	}

	refracted := Refract(ray.Direction(), outwardNormal, niOverNt)
	if refracted != nil {
		reflectProb = Schlick(cosine, dielectric.refractiveIndex)
	} else {
		reflectProb = 1.0
	}
	if rand.Float64() < reflectProb {
		scatter = NewRay(hitRecord.p, reflected)
	} else {
		scatter = NewRay(hitRecord.p, refracted)
	}
	return
}

type Lambertian struct {
	albedo *Vec
}

func NewLambertian(albedo *Vec) *Lambertian {
	return &Lambertian{
		albedo: albedo,
	}
}

func (lambertian *Lambertian) Scatter(ray *Ray, hitRecord *HitRecord) (scatter *Ray, attenuation *Vec) {
	target := hitRecord.p.Add(hitRecord.normal).Add(RandomInUnitSphere())
	scattered := NewRay(hitRecord.p, target.Sub(hitRecord.p))
	return scattered, lambertian.albedo
}

type Hittable interface {
	Hit(ray *Ray, tmin float64, tmax float64) *HitRecord
}

type Vec struct {
	x float64
	y float64
	z float64
}

func (v *Vec) X() float64 {
	return v.x
}

func (v *Vec) Y() float64 {
	return v.y
}

func (v *Vec) Z() float64 {
	return v.z
}

func (v *Vec) Len() float64 {
	return math.Sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
}

func (v *Vec) Neg() *Vec {
	return NewVec(-v.x, -v.y, -v.z)
}

func (v *Vec) Add(v1 *Vec) *Vec {
	return NewVec(
		v.X()+v1.X(),
		v.Y()+v1.Y(),
		v.Z()+v1.Z(),
	)
}

func (v Vec) Sub(v1 *Vec) *Vec {
	return NewVec(
		v.X()-v1.X(),
		v.Y()-v1.Y(),
		v.Z()-v1.Z(),
	)
}

func (v *Vec) Mul(v1 *Vec) *Vec {
	return NewVec(
		v.X()*v1.X(),
		v.Y()*v1.Y(),
		v.Z()*v1.Z(),
	)
}
func (v *Vec) MulConst(c float64) *Vec {
	return NewVec(
		c*v.X(),
		c*v.Y(),
		c*v.Z(),
	)
}
func (v *Vec) DivConst(c float64) *Vec {
	return NewVec(
		v.X()/c,
		v.Y()/c,
		v.Z()/c,
	)
}

func Cross(v1, v2 *Vec) *Vec {
	return NewVec(
		v1.y*v2.z-v1.z*v2.y,
		-v1.x*v2.z+v1.z*v2.x,
		v1.x*v2.y-v1.y*v2.x,
	)
}

func (v *Vec) Unit() *Vec {
	length := v.Len()
	return NewVec(v.x/length, v.y/length, v.z/length)
}

func (v *Vec) Dot(b *Vec) float64 {
	return v.x*b.x + v.y*b.y + v.z*b.z
}

func (v *Vec) String() string {
	return fmt.Sprintf("x: %f, y: %f, z: %f, len: %f", v.x, v.y, v.z, v.Len())
}

func PrintToPPMFile(v *Vec, w io.Writer) {
	unit := v
	r, g, b := int(255.99*unit.x), int(255.99*unit.y), int(255.99*unit.z)
	fmt.Fprintf(w, "%d %d %d\n", r, g, b)
}

type Ray struct {
	origin    *Vec
	direction *Vec
}

var (
	rayPool = &sync.Pool{}
)

func putRay(ray *Ray) {
	ray.direction = nil
	ray.origin = nil
	rayPool.Put(ray)
}

func NewRay(origin *Vec, direction *Vec) *Ray {
	obj := rayPool.Get()
	if obj != nil {
		ray := obj.(*Ray)
		ray.direction = direction
		ray.origin = origin
		return ray
	}
	return &Ray{
		origin:    origin,
		direction: direction,
	}
}

func (r *Ray) Origin() *Vec {
	return r.origin
}

func (r *Ray) Direction() *Vec {
	return r.direction
}

func (r *Ray) PointAtParameter(t float64) *Vec {
	return r.origin.Add(r.direction.MulConst(t))
}

func Color(r *Ray, hittable Hittable, recursionDepth int) *Vec {
	hr := hittable.Hit(r, 0.001, math.MaxFloat64)

	// HIT!
	if hr != nil {
		scatterRay, attenuation := hr.material.Scatter(r, hr)
		putHitRecord(hr)
		if scatterRay == nil || recursionDepth > 10 {
			return NewVec(0, 0, 0)
		}
		color := Color(scatterRay, hittable, recursionDepth+1).Mul(attenuation)
		putRay(scatterRay)
		return color
	}

	unitDirection := r.direction.Unit()
	t := 0.5 * (unitDirection.y + 1)
	return NewVec(1, 1, 1).MulConst(1 - t).Add(NewVec(0.5, 0.7, 1).MulConst(t))
}

func NewVec(x, y, z float64) *Vec {
	return &Vec{
		x: x,
		y: y,
		z: z,
	}
}

type Hittables struct {
	List []Hittable
}

func (hittables *Hittables) Hit(ray *Ray, tmin float64, tmax float64) *HitRecord {
	closestT := tmax
	hitAnything := false
	var solHr *HitRecord
	for _, hittable := range hittables.List {
		hr := hittable.Hit(ray, tmin, closestT)
		if hr == nil {
			continue
		}
		hitAnything = true
		closestT = hr.t
		solHr = hr
	}
	if !hitAnything {
		return nil
	}
	return solHr
}

type Sphere struct {
	center   *Vec
	radius   float64
	material Material
}

func NewSphere(center *Vec, radius float64, material Material) *Sphere {
	return &Sphere{
		center:   center,
		radius:   radius,
		material: material,
	}
}

var (
	hitRecordPool = &sync.Pool{}
)

func getHitRecord() *HitRecord {
	obj := hitRecordPool.Get()
	if obj == nil {
		return &HitRecord{}
	}

	return obj.(*HitRecord)
}

func putHitRecord(hr *HitRecord) {
	hr.material = nil
	hr.p = nil
	hr.normal = nil
	hitRecordPool.Put(hr)
}

func (sphere *Sphere) Hit(ray *Ray, tmin float64, tmax float64) *HitRecord {
	oc := ray.origin.Sub(sphere.center)
	a := ray.direction.Dot(ray.direction)
	b := oc.Dot(ray.direction)
	c := oc.Dot(oc) - sphere.radius*sphere.radius
	discriminant := b*b - a*c
	if discriminant < 0 {
		return nil
	}

	hr := getHitRecord()
	hr.material = sphere.material
	tmp := (-b - math.Sqrt(discriminant)) / a
	if tmin < tmp && tmp < tmax {
		hr.t = tmp
		hr.p = ray.PointAtParameter(tmp)
		hr.normal = (hr.p.Sub(sphere.center)).DivConst(sphere.radius)
		return hr
	}
	tmp = (-b + math.Sqrt(b*b-a*c)) / a

	if tmin < tmp && tmp < tmax {
		hr.t = tmp
		hr.p = ray.PointAtParameter(tmp)
		hr.normal = (hr.p.Sub(sphere.center)).DivConst(sphere.radius)
		return hr
	}
	return nil
}

type Camera struct {
	origin          *Vec
	lowerLeftCorner *Vec
	horizontal      *Vec
	vertical        *Vec
	u               *Vec
	w               *Vec
	v               *Vec
	lensRadius      float64
}

func RandomInUnitDisk() *Vec {
	for {
		p := NewVec(rand.Float64(), rand.Float64(), 0).Sub(NewVec(1, 1, 0))
		if p.Dot(p) >= 1.0 {
			return p
		}
	}
}

func NewCamera(lookFrom *Vec, lookAt *Vec, vup *Vec, vfow float64, aspect float64, aperture float64, focusDistance float64) *Camera {
	lensRadius := aperture / 2.0
	theta := vfow * math.Pi / 180.0
	halfHeight := math.Tan(theta / 2.0)
	halfWidth := aspect * halfHeight

	w := lookFrom.Sub(lookAt).Unit()
	u := Cross(vup, w).Unit()
	v := Cross(w, u)

	return &Camera{
		origin:          lookFrom,
		lowerLeftCorner: lookFrom.Sub(u.MulConst(halfWidth * focusDistance)).Sub(v.MulConst(halfHeight * focusDistance)).Sub(w.MulConst(focusDistance)),
		horizontal:      u.MulConst(2 * halfWidth * focusDistance),
		vertical:        v.MulConst(2 * halfHeight * focusDistance),
		lensRadius:      lensRadius,
		u:               u,
		w:               w,
		v:               v,
	}
}

func (camera *Camera) GetRay(u, v float64) *Ray {
	rd := RandomInUnitDisk().MulConst(camera.lensRadius)
	offset := camera.u.MulConst(rd.X()).Add(camera.v.MulConst(rd.Y()))
	return NewRay(camera.origin.Add(offset), camera.lowerLeftCorner.Add(camera.horizontal.MulConst(u).Add(camera.vertical.MulConst(v))).Sub(camera.origin).Sub(offset))
}

func RandomInUnitSphere() *Vec {
	var vec *Vec
	for {
		vec = NewVec(rand.Float64(), rand.Float64(), rand.Float64()).MulConst(2).Sub(NewVec(1, 1, 1))
		if vec.Dot(vec) < 1 {
			break
		}
	}
	return vec
}

func RandomWorld() *Hittables {
	world := &Hittables{}
	world.List = append(world.List, NewSphere(NewVec(0, -1000, 0), 1000, NewLambertian(NewVec(0.5, 0.5, 0.5))))

	for a := -11; a < 11; a++ {
		for b := -11; b < 11; b++ {
			mat := rand.Float64()
			center := NewVec(float64(a)+0.9*rand.Float64(), 0.2, float64(b)+0.9*rand.Float64())
			if center.Sub(NewVec(4, 0.2, 0)).Len() > 0.9 {
				switch {
				case mat < 0.8:
					world.List = append(world.List, NewSphere(center, 0.2, NewLambertian(NewVec(rand.Float64(), rand.Float64(), rand.Float64()))))
				case mat < 0.95:
					world.List = append(world.List, NewSphere(center, 0.2, NewMetal(NewVec(0.5*(1+rand.Float64()), 0.5*(1+rand.Float64()), 0.5*(1+rand.Float64())), 0.5*rand.Float64())))
				default:
					world.List = append(world.List, NewSphere(center, 0.2, NewDielectric(1.5)))
				}
			}
		}
	}
	world.List = append(world.List, NewSphere(NewVec(0, 1, 0), 1, NewDielectric(1.5)))
	world.List = append(world.List, NewSphere(NewVec(-4, 1, 0), 1, NewLambertian(NewVec(0.4, 0.2, 0.1))))
	world.List = append(world.List, NewSphere(NewVec(4, 1, 0), 1, NewMetal(NewVec(0.7, 0.6, 0.5), 0)))

	return world
}

func run() {

	cfg := pixelgl.WindowConfig{
		Title:  "Path tracing",
		Bounds: pixel.R(0, 0, WIDTH, HEIGHT),
		VSync:  true,
	}

	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		panic(err)
	}
	win.Clear(colornames.Black)
	hittables := RandomWorld()
	go func() {
		drawOnWindow(win, hittables, 0)
		drawOnWindow(win, hittables, 1)
		drawOnWindow(win, hittables, 4)
		drawOnWindow(win, hittables, 8)
		drawOnWindow(win, hittables, 16)
		drawOnWindow(win, hittables, 32)
		drawOnWindow(win, hittables, 128)
	}()
	for !win.Closed() {
		win.Update()
	}
}

func drawOnWindow(win *pixelgl.Window, hittables Hittable, AntiAliasSample int) {

	canvas := win.Canvas()
	pixels := canvas.Pixels()

	lookFrom := NewVec(24, 4, -10)
	lookAt := NewVec(0, 0, 0)
	aperture := 2.0
	distanceToFocus := lookFrom.Sub(lookAt).Len()
	camera := NewCamera(lookFrom, lookAt, NewVec(0, 1, 0), 10, WIDTH/HEIGHT, aperture, distanceToFocus)
	//redColor := NewVec(1, 0, 0)
	wg := &sync.WaitGroup{}
	drawRow := func(jStart, jEnd int) {
		for j := jStart; j < jEnd; j++ {
			for i := 0; i < WIDTH; i++ {
				color := NewVec(0, 0, 0)
				for k := 0; k < AntiAliasSample; k++ {

					u := float64(float64(i)+rand.Float64()) / float64(WIDTH)
					v := float64(float64(j)+rand.Float64()) / float64(HEIGHT)
					ray := camera.GetRay(u, v)

					//p := ray.PointAtParameter(2.0)
					color = color.Add(Color(ray, hittables, 0))
					putRay(ray)
				}
				color = color.DivConst(float64(AntiAliasSample))
				// gamma corrections
				color = NewVec(math.Sqrt(color.X()), math.Sqrt(color.Y()), math.Sqrt(color.Z()))

				r, g, b, a := uint8(255*color.X()), uint8(255*color.Y()), uint8(255*color.Z()), uint8(255)
				offset := 4 * ((j * HEIGHT) + i)
				pixels[offset+0] = r
				pixels[offset+1] = g
				pixels[offset+2] = b
				pixels[offset+3] = a
			}
		}
		wg.Done()
		canvas.SetPixels(pixels)
	}

	type job struct {
		jStart int
		jEnd   int
	}

	jobCh := make(chan job)
	drawer := func() {
		for {
			j, ok := <-jobCh
			if !ok {
				return
			}
			drawRow(j.jStart, j.jEnd)
		}
	}

	for i := 0; i < 4; i++ {
		go drawer()
	}

	for j := HEIGHT - 1; j >= 0; j -= RowsPerGoroutine {
		min := j - RowsPerGoroutine
		if min < 0 {
			min = 0
		}
		wg.Add(1)
		jobCh <- job{min, j}
	}
	wg.Wait()
	canvas.SetPixels(pixels)
	close(jobCh)
}

func main() {
	rand.Seed(5)
	pixelgl.Run(run)
}
