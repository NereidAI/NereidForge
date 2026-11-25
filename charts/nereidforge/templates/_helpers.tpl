{{/*
Copyright (C) 2026 YIQISOFT
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "nereidforge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nereidforge.fullname" -}}
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

{{- define "nereidforge.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nereidforge.labels" -}}
helm.sh/chart: {{ include "nereidforge.chart" . }}
{{ include "nereidforge.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nereidforge.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nereidforge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "nereidforge.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nereidforge.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "nereidforge.runtimeSecretName" -}}
{{- .Values.runtimeSecret.existingSecret | default (printf "%s-secrets" (include "nereidforge.fullname" .)) -}}
{{- end }}

{{- define "nereidforge.api.fullname" -}}
{{- if .Values.api.nameOverride }}
{{- .Values.api.nameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-api" (include "nereidforge.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "nereidforge.worker.fullname" -}}
{{- if .Values.worker.nameOverride }}
{{- .Values.worker.nameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-worker" (include "nereidforge.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- define "nereidforge.frontend.fullname" -}}
{{- if .Values.frontend.nameOverride }}
{{- .Values.frontend.nameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-frontend" (include "nereidforge.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
