# API de serviço (Golang). Build multi-stage: compila estático, roda em
# imagem mínima sem toolchain Go.
FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /out/api .

FROM gcr.io/distroless/base-debian12
COPY --from=build /out/api /api
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/api"]
