# Slim-Gest Helm Chart

This directory contains the Helm chart for deploying the slim-gest PDF OCR service to Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- NVIDIA GPU operator or device plugin installed in your cluster
- GPU-enabled nodes in your cluster
- Models directory accessible to the cluster (via hostPath or PersistentVolume)

## Quick Start

### 1. Build the Docker Image

```bash
cd /home/jdyer/Development/slim-gest
docker build -t slim-gest:latest .
```

If you're using a container registry:

```bash
docker tag slim-gest:latest your-registry/slim-gest:latest
docker push your-registry/slim-gest:latest
```

### 2. Configure Values

Edit `slim-gest/values.yaml` to configure your deployment:

**Required Changes:**

```yaml
# Update the models path to match your setup
volumes:
  models:
    hostPath:
      path: /path/to/your/models  # Change this!
```

**Optional Changes:**

```yaml
# If using a container registry
image:
  repository: your-registry/slim-gest
  tag: "latest"

# Adjust resource limits based on your GPU
resources:
  limits:
    nvidia.com/gpu: 1
    memory: 8Gi
  requests:
    nvidia.com/gpu: 1
    memory: 4Gi
    cpu: 2000m
```

### 3. Install the Chart

```bash
# From the helm directory
cd slim-gest

# Install with default values
helm install slim-gest .

# Or install with custom values
helm install slim-gest . -f custom-values.yaml

# Install in a specific namespace
helm install slim-gest . --namespace ocr-service --create-namespace
```

### 4. Verify the Deployment

```bash
# Check the deployment status
kubectl get deployments
kubectl get pods

# Check the service
kubectl get services

# View logs
kubectl logs -l app.kubernetes.io/name=slim-gest
```

### 5. Access the Service

**Port Forward (for testing):**

```bash
kubectl port-forward svc/slim-gest 7670:7670
```

Then access:
- API: http://localhost:7670/
- Swagger UI: http://localhost:7670/docs
- ReDoc: http://localhost:7670/redoc

**NodePort (for external access):**

```bash
# Change service type in values.yaml
service:
  type: NodePort

# Upgrade the deployment
helm upgrade slim-gest .

# Get the NodePort
kubectl get svc slim-gest
```

**LoadBalancer (cloud environments):**

```bash
# Change service type in values.yaml
service:
  type: LoadBalancer

# Upgrade the deployment
helm upgrade slim-gest .

# Get the external IP
kubectl get svc slim-gest
```

**Ingress (recommended for production):**

```yaml
# In values.yaml
ingress:
  enabled: true
  className: nginx  # or your ingress controller
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: ocr.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: slim-gest-tls
      hosts:
        - ocr.yourdomain.com
```

## Configuration

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Docker image repository | `slim-gest` |
| `image.tag` | Docker image tag | `latest` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `7670` |
| `resources.limits.nvidia.com/gpu` | Number of GPUs | `1` |
| `volumes.models.hostPath.path` | Path to models directory | `/path/to/models` |
| `ingress.enabled` | Enable ingress | `false` |

### GPU Configuration

The chart requests NVIDIA GPUs by default. Ensure your cluster has:

1. **NVIDIA Device Plugin** installed:
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
   ```

2. **GPU nodes labeled** (usually automatic):
   ```bash
   kubectl get nodes -o json | jq '.items[].status.allocatable'
   ```

3. **Node selector or affinity** (optional) to target GPU nodes:
   ```yaml
   nodeSelector:
     accelerator: nvidia-gpu
   ```

### Volume Configuration

The models must be accessible to the pods. Options:

**1. HostPath (simplest for single-node or shared filesystem):**

```yaml
volumes:
  models:
    type: hostPath
    hostPath:
      path: /mnt/models/slim-gest
      type: Directory
```

**2. PersistentVolumeClaim (recommended for production):**

```yaml
volumes:
  models:
    type: persistentVolumeClaim
    persistentVolumeClaim:
      claimName: slim-gest-models-pvc
```

Then create a PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: slim-gest-models-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: your-storage-class
```

## Upgrading

```bash
# Upgrade with new values
helm upgrade slim-gest . -f values.yaml

# Upgrade with new image
helm upgrade slim-gest . --set image.tag=v0.2.0
```

## Uninstalling

```bash
# Uninstall the release
helm uninstall slim-gest

# Uninstall from specific namespace
helm uninstall slim-gest --namespace ocr-service
```

## Testing

### Test the Deployment

```bash
# Port forward to the service
kubectl port-forward svc/slim-gest 7670:7670 &

# Test health endpoint
curl http://localhost:7670/

# Test OCR processing
curl -X POST http://localhost:7670/process-pdf \
  -F "file=@test.pdf" \
  -F "dpi=150"
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=slim-gest

# View pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

### GPU Not Available

```bash
# Check if GPUs are available in the cluster
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Check if NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia
```

### Models Not Loading

```bash
# Check if volume is mounted correctly
kubectl exec <pod-name> -- ls -la /app/models

# Verify environment variable
kubectl exec <pod-name> -- env | grep NEMOTRON
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints slim-gest

# Check if pod is ready
kubectl get pods -l app.kubernetes.io/name=slim-gest

# Test from within cluster
kubectl run curl-test --image=curlimages/curl:latest --rm -it -- sh
# Then: curl http://slim-gest:7670/
```

## Production Considerations

### Resource Management

- Set appropriate resource requests and limits based on your workload
- Monitor GPU utilization and adjust replicas accordingly
- Consider using HPA (Horizontal Pod Autoscaler) for CPU-based scaling

### Security

- Use non-root containers if possible
- Enable Pod Security Standards
- Use network policies to restrict traffic
- Store sensitive configuration in Secrets

### Monitoring

- Integrate with Prometheus for metrics
- Set up alerts for GPU utilization, memory usage, and pod health
- Monitor API latency and error rates

### High Availability

- Run multiple replicas across different nodes (if you have multiple GPUs)
- Use pod anti-affinity to spread pods
- Configure appropriate readiness and liveness probes

## Examples

See the `examples/` directory for sample configurations:
- `examples/production-values.yaml` - Production-ready configuration
- `examples/ingress-setup.yaml` - Ingress configuration with TLS
- `examples/pvc-setup.yaml` - PersistentVolumeClaim configuration

## Support

For issues and questions:
- Check the main project README at `/home/jdyer/Development/slim-gest/README.md`
- Review the web service documentation at `/home/jdyer/Development/slim-gest/src/slimgest/web/README.md`
