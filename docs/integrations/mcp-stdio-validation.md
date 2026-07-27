# MCP Stdio Validation

## Validated Boundary

On 2026-07-27, the official Python MCP SDK `ClientSession` version 1.28.1 launched
`python -m diw.mcp_server` as a separate stdio process and completed this transcript:

1. Initialized the MCP session and negotiated protocol version `2025-11-25`.
2. Discovered exactly `search_documents` and `get_research_record`.
3. Confirmed neither tool schema accepts `tenant_id`.
4. Invoked evidence search and received only the configured tenant's document.
5. Looked up the configured tenant's research record successfully.
6. Looked up another tenant's record and received `{"found": false}`.
7. Supplied another tenant ID as an extra model argument; the server remained pinned to its
   process-configured tenant and did not return the other tenant's evidence.

The server exposed no write tools. The credential-free transcript is
[`mcp-stdio-validation.json`](../../results/evidence/mcp-stdio-validation.json).

## Client Configuration

[`mcp-client.example.json`](../../configs/mcp-client.example.json) records the external client
shape with absolute-path placeholders and no secrets. The tenant is owned by process
configuration, not model input.

## Reproduce

Install the MCP extra and run the validation:

```bash
python -m pip install -e ".[mcp]"
make validate-mcp-stdio
```

The validator creates a temporary two-tenant database, launches the server as a child process,
performs discovery and calls through the SDK client, asserts the isolation boundary, and saves the
transcript.

## Honest Limitation

This validates an external client process over local stdio. It does not claim a remotely hosted
MCP transport, OAuth for MCP, or write-capable MCP tools.

## Validated Scope

> Built a tenant-pinned, read-only MCP stdio server and validated tool discovery, evidence search,
> record lookup, and cross-tenant denial through an external MCP client.
