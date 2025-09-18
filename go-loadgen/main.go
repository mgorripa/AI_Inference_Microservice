package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"time"

	prom "github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type predictIn struct {
	X []float32 `json:"x"`
}

var (
	reqTotal = prom.NewCounterVec(
		prom.CounterOpts{Name: "loadgen_requests_total", Help: "total requests"},
		[]string{"endpoint", "status"},
	)
	latency = prom.NewHistogramVec(
		prom.HistogramOpts{Name: "loadgen_latency_seconds", Help: "request latency", Buckets: prom.DefBuckets},
		[]string{"endpoint"},
	)
)

func hitPredict(url string, client *http.Client) {
	// Create a random input vector of length 16 in range [-1, 1]
	vec := make([]float32, 16)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1
	}

	body, _ := json.Marshal(predictIn{X: vec})
	start := time.Now()
	resp, err := client.Post(url+"/predict", "application/json", bytes.NewReader(body))
	latency.WithLabelValues("predict").Observe(time.Since(start).Seconds())

	if err != nil {
		reqTotal.WithLabelValues("predict", "err").Inc()
		return
	}
	defer resp.Body.Close()

	reqTotal.WithLabelValues("predict", fmt.Sprint(resp.StatusCode)).Inc()
}

func main() {
	prom.MustRegister(reqTotal, latency)

	service := getenv("SERVICE_URL", "http://ai-infer-service:8080")
	qps := getenvInt("QPS", 2)
	concurrency := getenvInt("CONCURRENCY", 2)

	http.Handle("/metrics", promhttp.Handler())
	go func() {
		log.Println("Prometheus metrics available on :9090/metrics")
		if err := http.ListenAndServe(":9090", nil); err != nil {
			log.Fatalf("metrics server error: %v", err)
		}
	}()

	client := &http.Client{Timeout: 5 * time.Second}
	ticker := time.NewTicker(time.Second)
	sem := make(chan struct{}, concurrency)

	for range ticker.C {
		for i := 0; i < qps; i++ {
			sem <- struct{}{}
			go func() {
				hitPredict(service, client)
				<-sem
			}()
		}
	}
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func getenvInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		var x int
		fmt.Sscanf(v, "%d", &x)
		return x
	}
	return def
}
