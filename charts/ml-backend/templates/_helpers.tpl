{{/*
Copyright (C) 2026 YIQISOFT
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "ml-backend.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "ml-backend.fullname" -}}
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

{{- define "ml-backend.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "ml-backend.labels" -}}
helm.sh/chart: {{ include "ml-backend.chart" . }}
{{ include "ml-backend.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "ml-backend.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-backend.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "ml-backend.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ml-backend.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
