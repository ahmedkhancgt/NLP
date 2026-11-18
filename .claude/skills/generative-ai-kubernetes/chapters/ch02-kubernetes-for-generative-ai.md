# Chapter 2: Kubernetes for Generative AI

## Core Idea
Kubernetes' control-plane/data-plane separation, declarative control loops, autoscaling (HPA/VPA/Cluster Autoscaler), namespace-based isolation, and GPU device-plugin extensibility together provide the resilient, scalable, secure substrate that GenAI/ML pipelines are built on top of.

## Frameworks Introduced
- **Control plane / data plane separation**: control nodes run the API server, scheduler, controllers, and etcd; worker (data plane) nodes run Pods/containers via kubelet.
  - When to use: always — this is the architectural foundation of resilience; in production, separate these onto independent hosts.
  - How: control nodes host API server + scheduler + controller-manager + etcd; workers register with kubelet and run scheduled workloads.
- **Three-tier autoscaling**: Horizontal Pod Autoscaler (HPA, adds pod replicas by metric e.g. CPU), Vertical Pod Autoscaler (VPA, raises CPU/memory limits per pod), Cluster Autoscaler (adds/removes nodes).
  - When to use: HPA for stateless scale-out under load; VPA when individual pods need more resources than scale-out solves; Cluster Autoscaler when the whole cluster is resource-constrained.
  - How: `kubectl autoscale deployment <name> --cpu-percent=<N> --min=<N> --max=<N>` or apply an HPA YAML referencing a metrics source (e.g. metrics-server).
- **Device Plugin Framework (GPU support)**: lets Kubernetes schedule/limit/distribute vendor-specific hardware (GPUs) via a DaemonSet-deployed plugin.
  - When to use: any cluster running GPU-backed training/inference workloads.
  - How: install vendor GPU drivers + container toolkit on nodes → deploy device plugin DaemonSet (e.g. `nvidia-device-plugin`) → request `nvidia.com/gpu` in pod resource limits + add tolerations for GPU-tainted nodes.
- **Namespace isolation + ResourceQuotas**: namespaces group resources; ResourceQuotas enforce hard caps (CPU, memory, extended resources like GPUs) per namespace; requests vs. limits define minimum guaranteed vs. maximum allowed.
  - When to use: multi-team/multi-project clusters needing resource partitioning and basic governance.
  - How: `kubectl create ns <name>`, then apply a `ResourceQuota` manifest with `hard:` limits including `requests.nvidia.com/gpu`.

## Key Concepts
- **Pod**: one or more containers scheduled together as the atomic unit of deployment.
- **Deployment**: manages Pod lifecycle (create/update/scale/remove) and desired replica count.
- **Service**: stable network identity/abstraction over a dynamic set of Pods.
- **Controller / control loop**: continuously reconciles actual state toward desired state (core to K8s resilience).
- **Probes (startup/liveness/readiness)**: health checks that gate restarts and traffic routing.
- **Extended Resources**: non-CPU/memory resources (e.g. GPUs) requestable since K8s 1.10; limits and requests are treated identically for these.
- **Operator pattern / CRD**: community-defined custom resource types (e.g. GPUs) plus a controller that manages their lifecycle beyond core K8s primitives.
- **Secrets**: in-cluster encrypted-at-rest storage for sensitive data (passwords, keys), decoupled from app config, supporting third-party backends (Vault, cloud secret managers).

## Mental Models
- Think of the **control loop** as "Kubernetes never stops asking 'are we in the desired state?'" — this is why failures self-heal and scaling is automatic rather than manual.
- Use **containers = app portability, Kubernetes = deployment/ops portability** — containers abstract the app from the OS; K8s abstracts *how you run and manage* that app across infrastructure.
- Think of **namespaces as grouping, not real isolation** — true multi-tenancy requires network policies, storage plugins, and RBAC layered on top; namespaces alone just organize and allow quota enforcement.
- GPU scheduling mental model: **GPUs are a schedulable, quota-able resource just like CPU/memory**, but currently with far less efficient sharing (see Ch4's GPU-sharing table) — plan capacity conservatively.

## Anti-patterns
- **Treating namespaces as a security boundary by themselves**: namespaces provide grouping/quota, not network or storage isolation — a false sense of tenant isolation is a common misconfiguration.
- **Running out of GPU without quotas**: without `ResourceQuota` on `requests.nvidia.com/gpu`, one namespace/team can monopolize all cluster GPUs.
- **Exposing SSH (port 22) or demo ports to "Any Source" in cloud NSGs**: the book explicitly flags this as an attack-prone misconfiguration — restrict inbound rules to known IPs.
- **Setting hard memory limits without understanding OOM kill behavior**: exceeding a memory limit causes Kubernetes to OOM-kill the pod — many teams therefore only set `requests`, not `limits`, for memory-sensitive ML workloads.

## Code Examples
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hello-scaling
spec:
  selector:
    matchLabels:
      app: hello-scaling
  template:
    metadata:
      labels:
        app: hello-scaling
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: "kubernetes.io/hostname"
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: hello-scaling
      containers:
      - name: php-apache
        image: registry.k8s.io/hpa-example
        ports:
        - name: http
          containerPort: 80
        resources:
          limits:
            cpu: 500m
          requests:
            cpu: 200m
        terminationGracePeriodSeconds: 1
```
- **What it demonstrates**: `topologySpreadConstraints` spreads replicas across hosts for resiliency; `resources.limits/requests` caps CPU for scheduler-aware bin-packing — both patterns reused throughout the book's GPU and HPA examples.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  restartPolicy: Never
  containers:
    - name: cuda-container
      image: nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
      resources:
        requests:
          cpu: "1"
          memory: "1Gi"
        limits:
          nvidia.com/gpu: 1
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```
- **What it demonstrates**: requesting a GPU via `limits.nvidia.com/gpu` and tolerating the GPU-node taint so the pod schedules only on GPU-capable nodes.

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: single-gpu-quota
spec:
  hard:
    requests.cpu: "1"
    requests.memory: 1Gi
    requests.nvidia.com/gpu: 1
```
- **What it demonstrates**: namespace-scoped GPU quota enforcement (a sibling `no-gpus-quota` sets `requests.nvidia.com/gpu: 0` to block GPU use in `default`).

## Reference Tables
| Autoscaler | Scales | Trigger |
|---|---|---|
| Horizontal Pod Autoscaler (HPA) | Number of Pod replicas | Average metric (e.g. CPU%) |
| Vertical Pod Autoscaler (VPA) | CPU/Memory limits of individual Pods | Resource pressure |
| Cluster Autoscaler | Number of cluster nodes | Pending pods / low node resources |

## Worked Example
**Hands-on HPA scaling walkthrough:** Deploy `hello-scaling` (Deployment) + `hello-scaling-service` (NodePort Service) to minikube → create HPA with `kubectl autoscale deployment hello-scaling --cpu-percent=50 --min=1 --max=6` → deploy a load-generator Deployment (3 replicas of `busybox` running `while true; do wget -q -O- http://hello-scaling-svc; done`) → watch `kubectl get hpa hello-scaling --watch` show CPU utilization climb above 50% and replica count increase → delete the load deployment and watch replicas scale back down to 1. This demonstrates the full reconciliation loop: metrics-server feeds CPU%, HPA controller adjusts replica count, Deployment controller creates/removes Pods.

**GPU-enabled minikube walkthrough:** provision an Azure NC4as_T4_v3 VM → install NVIDIA GPU Driver Extension (installs drivers + CUDA toolkit) → install Docker + NVIDIA Container Toolkit, configure `/etc/docker/daemon.json` with `"default-runtime": "nvidia"` → `minikube start --driver docker --container-runtime docker --gpus all` → verify with `kubectl describe nodes` showing `nvidia.com/gpu` capacity → run the `gpu-pod` CUDA vector-add sample to confirm GPU visibility to Kubernetes.

## Key Takeaways
1. Kubernetes' control loop (desired-state reconciliation) is the single mechanism underlying resilience, scaling, and self-healing — understand it once, apply it everywhere.
2. GPUs are exposed to Kubernetes via the Device Plugin Framework and consumed as an `nvidia.com/gpu` extended resource — this is the building block for all later GPU-scheduling chapters.
3. Namespaces + ResourceQuotas give basic multi-tenant governance but are not a full isolation mechanism — pair with RBAC and NetworkPolicies (Ch10) for real separation.
4. HPA/VPA/Cluster Autoscaler operate at different layers (pod count, pod size, node count) — production designs often combine more than one.
5. Kubernetes Secrets, RBAC, and policy/admission controls (introduced briefly here) are the entry point to the deeper security chapter (Ch10).

## Connects To
- **Ch1**: directly implements the "Infrastructure needs for generative AI" (GPUs, storage, network) this chapter closes with.
- **Ch3-4**: reuse minikube + GPU setup patterns established here for hands-on foundational model deployment.
- **Ch10**: expands namespace/RBAC/Secrets concepts into a full production security model.
