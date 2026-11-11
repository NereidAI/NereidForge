# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# deploy-tools: tool image used by the kubeflow chart's install Job.
# Ships kubectl + kustomize + python3 so files/install-kubeflow.sh can run
# (kustomize build with pristine public image references, then kubectl apply).
FROM alpine:3.24.4

ARG KUBECTL_VERSION=v1.29.0
ARG KUSTOMIZE_VERSION=v5.3.0

RUN apk add --no-cache bash curl tar ca-certificates python3 py3-pip

# kubectl
RUN curl -fsSLo /usr/local/bin/kubectl \
      "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl

# kustomize (standalone, as used by files/install-kubeflow.sh: `kustomize build`)
RUN curl -fsSLo /tmp/kustomize.tar.gz \
      "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2F${KUSTOMIZE_VERSION}/kustomize_${KUSTOMIZE_VERSION}_linux_amd64.tar.gz" \
    && tar -xzf /tmp/kustomize.tar.gz -C /usr/local/bin \
    && chmod +x /usr/local/bin/kustomize \
    && rm -f /tmp/kustomize.tar.gz

CMD ["/bin/bash"]
