package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	requestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "loadgen_requests_total",
			Help: "Total number of requests sent by the load generator.",
		},
		[]string{"result", "status_code"},
	)

	requestDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "loadgen_request_duration_seconds",
			Help:    "End-to-end HTTP request duration observed by the load generator.",
			Buckets: prometheus.DefBuckets,
		},
	)

	inFlightRequests = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "loadgen_in_flight_requests",
			Help: "Number of requests currently waiting for a response.",
		},
	)
)

type config struct {
	targetURL       string
	qps             int
	concurrency     int
	duration        time.Duration
	requestTimeout  time.Duration
	metricsAddress  string
}

type predictRequest struct {
	X []float64 `json:"x"`
}

func main() {
	cfg := loadConfig()

	prometheus.MustRegister(
		requestsTotal,
		requestDuration,
		inFlightRequests,
	)

	// Cancel the process cleanly when Kubernetes or the user sends SIGTERM/SIGINT.
	ctx, stop := signal.NotifyContext(
		context.Background(),
		os.Interrupt,
		syscall.SIGTERM,
	)
	defer stop()

	// A duration of zero means "run until interrupted".
	if cfg.duration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.duration)
		defer cancel()
	}

	// Prometheus metrics are exposed separately from the traffic generator.
	metricsServer := &http.Server{
		Addr:              cfg.metricsAddress,
		Handler:           promhttp.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Printf("metrics available at http://0.0.0.0%s/metrics", cfg.metricsAddress)

		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("metrics server failed: %v", err)
		}
	}()

	log.Printf(
		"starting load generator: target=%s qps=%d concurrency=%d duration=%s timeout=%s",
		cfg.targetURL,
		cfg.qps,
		cfg.concurrency,
		cfg.duration,
		cfg.requestTimeout,
	)

	runLoad(ctx, cfg)

	shutdownContext, cancelShutdown := context.WithTimeout(
		context.Background(),
		5*time.Second,
	)
	defer cancelShutdown()

	if err := metricsServer.Shutdown(shutdownContext); err != nil {
		log.Printf("metrics server shutdown error: %v", err)
	}

	log.Println("load generator stopped")
}

func loadConfig() config {
	qps := readPositiveInt("QPS", 2)
	concurrency := readPositiveInt("CONCURRENCY", 2)
	durationSeconds := readNonNegativeInt("DURATION_SECONDS", 0)
	timeoutSeconds := readPositiveInt("REQUEST_TIMEOUT_SECONDS", 5)

	return config{
		targetURL:      readString("TARGET_URL", "http://localhost:8000/predict"),
		qps:            qps,
		concurrency:    concurrency,
		duration:       time.Duration(durationSeconds) * time.Second,
		requestTimeout: time.Duration(timeoutSeconds) * time.Second,
		metricsAddress: ":9090",
	}
}

func runLoad(ctx context.Context, cfg config) {
	client := &http.Client{
		Timeout: cfg.requestTimeout,
	}

	// The semaphore limits how many HTTP requests can be active simultaneously.
	semaphore := make(chan struct{}, cfg.concurrency)

	var workers sync.WaitGroup

	// Spread requests uniformly over one second instead of launching one burst.
	interval := time.Second / time.Duration(cfg.qps)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			workers.Wait()
			return

		case <-ticker.C:
			select {
			case semaphore <- struct{}{}:
				workers.Add(1)

				go func() {
					defer workers.Done()
					defer func() { <-semaphore }()

					sendPrediction(ctx, client, cfg.targetURL)
				}()

			default:
				// All concurrency slots are occupied. Record that this scheduled
				// request could not be launched instead of creating unbounded goroutines.
				requestsTotal.WithLabelValues("dropped", "not_sent").Inc()
			}
		}
	}
}

func sendPrediction(
	parentContext context.Context,
	client *http.Client,
	targetURL string,
) {
	payload := predictRequest{
		X: randomFeatures(16),
	}

	body, err := json.Marshal(payload)
	if err != nil {
		requestsTotal.WithLabelValues("encode_error", "not_sent").Inc()
		return
	}

	request, err := http.NewRequestWithContext(
		parentContext,
		http.MethodPost,
		targetURL,
		bytes.NewReader(body),
	)
	if err != nil {
		requestsTotal.WithLabelValues("request_creation_error", "not_sent").Inc()
		return
	}

	request.Header.Set("Content-Type", "application/json")

	startedAt := time.Now()
	inFlightRequests.Inc()

	response, err := client.Do(request)

	inFlightRequests.Dec()
	requestDuration.Observe(time.Since(startedAt).Seconds())

	if err != nil {
		requestsTotal.WithLabelValues("transport_error", "no_response").Inc()
		return
	}
	defer response.Body.Close()

	// Read and discard the response body so the HTTP connection can be reused.
	if _, err := io.Copy(io.Discard, response.Body); err != nil {
		requestsTotal.WithLabelValues(
			"response_read_error",
			strconv.Itoa(response.StatusCode),
		).Inc()
		return
	}

	result := "success"
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		result = "http_error"
	}

	requestsTotal.WithLabelValues(
		result,
		strconv.Itoa(response.StatusCode),
	).Inc()
}

func randomFeatures(size int) []float64 {
	values := make([]float64, size)

	for index := range values {
		// Generate finite values from a standard normal distribution.
		value := rand.NormFloat64()

		// This guard is defensive; rand.NormFloat64 normally returns finite values.
		if math.IsNaN(value) || math.IsInf(value, 0) {
			value = 0
		}

		values[index] = value
	}

	return values
}

func readString(name string, fallback string) string {
	value := os.Getenv(name)

	if value == "" {
		return fallback
	}

	return value
}

func readPositiveInt(name string, fallback int) int {
	value := os.Getenv(name)

	if value == "" {
		return fallback
	}

	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		log.Fatalf("%s must be a positive integer; received %q", name, value)
	}

	return parsed
}

func readNonNegativeInt(name string, fallback int) int {
	value := os.Getenv(name)

	if value == "" {
		return fallback
	}

	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 0 {
		log.Fatalf("%s must be a non-negative integer; received %q", name, value)
	}

	return parsed
}

func init() {
	// Use a process-specific seed for varied request payloads.
	rand.New(rand.NewSource(time.Now().UnixNano()))

	// Keep fmt imported for easy future console reporting and ensure compile-time use.
	_ = fmt.Sprintf
}
