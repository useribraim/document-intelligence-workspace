# Azure Deployment Notes

This is a target architecture for explaining Microsoft ecosystem awareness. It is not a claim of production deployment.

## Proposed Components

- Azure Container Apps or Azure App Service for FastAPI and Next.js services.
- Azure Database for PostgreSQL with pgvector support where available.
- Azure OpenAI Service for chat and embedding models.
- Azure Key Vault for secrets.
- Microsoft Entra ID for authentication.
- Microsoft 365 / SharePoint integration as a future document source.

## Security Notes

- Keep raw documents, generated outputs, and audit logs in separate tables.
- Avoid logging full sensitive source text in application logs.
- Store prompt/model/schema versions for AI-run reproducibility.
- Use per-user or per-workspace data boundaries.
- Prefer private networking and managed identity in a deployed environment.

## What To Build Locally First

Do not deploy before the local evidence loop is strong:

1. ingestion
2. hybrid retrieval
3. source-cited generation
4. validation
5. provenance
6. evaluation report
