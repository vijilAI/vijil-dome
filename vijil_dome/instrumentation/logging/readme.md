# OpenTelemetry Logging Components

## Resources
- Represents the entity producing the telemetry (e.g., your service).
- One Resource per LoggerProvider.
- Relationship: One-to-One with LoggerProvider.

## LoggerProvider
- Central point for creating Loggers and managing resources.
- Usually one global LoggerProvider per application.
- Relationship: 
  - One-to-Many with Loggers
  - One-to-One with Resource

## Processors
- Handle batching and export of log records.
- Multiple processors can be added to a LoggerProvider.
- Relationship: 
  - Many-to-One with LoggerProvider
  - One-to-One with Exporter

## Exporters
- Send the logs to their final destination (console, file, remote system).
- Multiple exporters can be used.
- Relationship: One-to-One with Processor

## Handlers
- Bridge between Python's logging system and OpenTelemetry.
- Multiple handlers can use the same LoggerProvider.
- Relationship: Many-to-One with LoggerProvider