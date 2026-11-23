{{/*
Copyright (C) 2026 YIQISOFT
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "nereidforge-backend.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nereidforge-backend.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "nereidforge-backend.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nereidforge-backend.labels" -}}
helm.sh/chart: {{ include "nereidforge-backend.chart" . }}
{{ include "nereidforge-backend.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nereidforge-backend.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nereidforge-backend.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "nereidforge-backend.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nereidforge-backend.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "nereidforge-backend.runtimeSecretName" -}}
{{- .Values.runtimeSecret.existingSecret | default (printf "%s-secrets" (include "nereidforge-backend.fullname" .)) -}}
{{- end }}

{{- define "nereidforge-backend.api.fullname" -}}
{{- if .Values.api.nameOverride }}
{{- .Values.api.nameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-api" (include "nereidforge-backend.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "nereidforge-backend.worker.fullname" -}}
{{- if .Values.worker.nameOverride }}
{{- .Values.worker.nameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-worker" (include "nereidforge-backend.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}