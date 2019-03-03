package main

import (
	"testing"
)

func BenchmarkColor(b *testing.B) {

	world := RandomWorld()

	lookFrom := NewVec(24, 4, -10)
	lookAt := NewVec(0, 0, 0)
	aperture := 1.0
	distanceToFocus := lookFrom.Sub(lookAt).Len()
	camera := NewCamera(lookFrom, lookAt, NewVec(0, 1, 0), 10, WIDTH/HEIGHT, aperture, distanceToFocus)

	u := float64(WIDTH/2) / float64(WIDTH)
	v := float64(HEIGHT/2) / float64(HEIGHT)
	ray := camera.GetRay(u, v)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Color(ray, world, 0)
	}
}
