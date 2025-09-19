# RESULTS

This section captures the **live outputs** from the working demo to show autoscaling and successful in‑cluster inference.

## HPA Status
```
$ kubectl get hpa
NAME               REFERENCE                     TARGETS         MINPODS   MAXPODS   REPLICAS   AGE
ai-infer-service   Deployment/ai-infer-service   cpu: 954%/20%   1         5         5          31m
```

## HPA Describe (excerpt)
```
Name:                                                  ai-infer-service
Namespace:                                             default
Reference:                                             Deployment/ai-infer-service
Metrics:                                               ( current / target )
  resource cpu on pods  (as a percentage of request):  884% (88m) / 20%
Min replicas:                                          1
Max replicas:                                          5
Deployment pods:                                       5 current / 5 desired
Conditions:
  AbleToScale     True    ScaleDownStabilized  recent recommendations were higher than current one, applying the highest recent recommendation
  ScalingActive   True    ValidMetricFound     the HPA was able to successfully calculate a replica count from cpu resource utilization (percentage of request)
  ScalingLimited  True    TooManyReplicas      the desired replica count is more than the maximum replica count
Events:
  Normal   SuccessfulRescale  New size: 4; reason: cpu resource utilization above target
  Normal   SuccessfulRescale  New size: 5; reason: cpu resource utilization above target
```

## Pods After Scaling
```
$ kubectl get pods -o wide
ai-infer-service-6df58c796-btjt6   1/1 Running   0   17m  10.244.0.9
ai-infer-service-6df58c796-c89cr   1/1 Running   0   17m  10.244.0.11
ai-infer-service-6df58c796-gxwb2   1/1 Running   0   17m  10.244.0.12
ai-infer-service-6df58c796-jjvmf   1/1 Running   0   16m  10.244.0.13
ai-infer-service-6df58c796-rx96p   1/1 Running   0   17m  10.244.0.10
ai-loadgen-7d8c7fbfcb-b6sxg        1/1 Running   0   5m   10.244.0.14
```

## In-Cluster API Checks
```
$ curl -s localhost:8080/healthz
{"ok":true,"cuda":false,"kernel":true}

$ curl -s -X POST localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}'
{"probs":[0.32863807678222656,0.6713619828224182]}
```

## Loadgen Metrics (Prometheus)
```
$ curl -s localhost:9090/metrics | head -n 20
# HELP go_gc_duration_seconds A summary of the pause duration of garbage collection cycles.
# TYPE go_gc_duration_seconds summary
...
```

> These results demonstrate: working inference service, custom C++ op loaded (`kernel:true`), and HPA scaling from 1 → 5 replicas under load.
