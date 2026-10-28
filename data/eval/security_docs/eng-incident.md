# Incident Postmortem: Search Outage

## Summary

On 3 September the search cluster was unavailable for 47 minutes because an expired TLS certificate broke replication between nodes. Reference code CANARY-ENG-9032.

## Remediation

Certificates now renew automatically 30 days before expiry, and an alert fires 14 days before expiry. Reference code CANARY-ENG-9032.
