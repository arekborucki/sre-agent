---
name: kubectl
description: Diagnosing Kubernetes problems (CrashLoopBackOff, OOMKilled, Pending, ImagePull, node NotReady, DNS/Service) with read-only commands.
---
# Skill: kubectl debugging

Best-practice playbook for diagnosing Kubernetes problems with read-only commands.

## General order
- Start wide with `kubectl get`, then `kubectl describe <kind> <name>` and read the
  **Events** section first. Most root causes show up there.
- `describe` before `logs`: Events explain *why a pod will not start*, logs explain
  *why the app crashed*.
- Recent cluster activity in order: `kubectl get events --sort-by=.lastTimestamp -n <ns>`.
- Always scope with `-n <namespace>`. Many "nothing found" mistakes are wrong-namespace.

## Pods
- **CrashLoopBackOff:** read the previous container with `kubectl logs <pod> --previous`
  (current logs are often empty after a restart). In `describe`, check Last State,
  exit code, and Reason (OOMKilled, Error).
- **OOMKilled / exit 137:** the container hit its memory limit. Compare `kubectl top pod`
  against `resources.limits.memory`. Fix is usually a higher limit or a memory leak.
- **Pending:** `describe pod` Events show why it cannot schedule (insufficient cpu/memory,
  no node matches nodeSelector/affinity, taints, or an unbound PVC).
- **ImagePullBackOff / ErrImagePull:** wrong image tag, private registry without an
  imagePullSecret, or the registry is unreachable.
- **Running but 0/1 Ready:** readiness probe is failing. Check the probe config and the
  endpoint it hits.

## Nodes
- **NotReady:** `kubectl describe node <node>` Conditions (MemoryPressure, DiskPressure,
  PIDPressure) and kubelet status. `kubectl get nodes -o wide` for context.
- Scheduling blocked by taints: `kubectl describe node <node> | grep Taints`.

## Networking and DNS
- Test DNS from inside the cluster, not the host. SERVFAIL or timeouts often mean CoreDNS
  is unhealthy or throttled: `kubectl -n kube-system get pods -l k8s-app=kube-dns`, check
  restarts and OOM.
- Service with no traffic: confirm `kubectl get endpoints <svc>` is non-empty. Empty means
  the selector matches no ready pods.

## Useful flags
- `-o wide` for node/IP context, `-o yaml` for the full spec, `--show-labels`.
- Find unhealthy pods fast: `kubectl get pods -A | grep -v Running`.

## Safety
- Stay read-only while diagnosing (get/describe/logs/top/events). Propose mutations as the
  fix; do not apply them without explicit approval.
