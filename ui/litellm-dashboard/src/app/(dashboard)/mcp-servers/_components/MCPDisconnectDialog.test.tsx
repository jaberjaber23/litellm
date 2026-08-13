import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import MCPDisconnectDialog from "./MCPDisconnectDialog";

const deleteMCPServerOAuthToken = vi.fn();
const deleteMCPOAuthUserCredential = vi.fn();

vi.mock("@/components/networking", () => ({
  deleteMCPServerOAuthToken: (...args: unknown[]) => deleteMCPServerOAuthToken(...args),
  deleteMCPOAuthUserCredential: (...args: unknown[]) => deleteMCPOAuthUserCredential(...args),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

const SHARED_BLAST_RADIUS = /Clears every OAuth token LiteLLM holds for this server, for every user/i;
const OWN_BLAST_RADIUS = /Clears only the token stored against your user/i;

function renderDialog(overrides: Partial<React.ComponentProps<typeof MCPDisconnectDialog>> = {}) {
  const onCleared = vi.fn();
  const onOpenChange = vi.fn();
  render(
    <MCPDisconnectDialog
      open
      mode="disconnect"
      serverId="srv-1"
      serverName="demo_server"
      accessToken="sk-test"
      isProxyAdmin
      onOpenChange={onOpenChange}
      onCleared={onCleared}
      {...overrides}
    />,
  );
  return { onCleared, onOpenChange };
}

describe("MCPDisconnectDialog blast radius", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    deleteMCPServerOAuthToken.mockResolvedValue({
      server_id: "srv-1",
      has_token: false,
      cleared: true,
      cleared_user_tokens: 2,
    });
    deleteMCPOAuthUserCredential.mockResolvedValue({ server_id: "srv-1", has_credential: false, is_expired: false });
  });

  it("states both blast radii, distinguishing the all-users clear from the caller's own connection", () => {
    renderDialog();

    const shared = screen.getByText(SHARED_BLAST_RADIUS);
    const own = screen.getByText(OWN_BLAST_RADIUS);

    expect(shared).toBeInTheDocument();
    expect(own).toBeInTheDocument();
    expect(shared.textContent).not.toEqual(own.textContent);
    expect(screen.getByText(/Every stored token for this server \(affects all users\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Only your own connection \(affects just you\)/i)).toBeInTheDocument();
  });

  it("warns that the all-users clear revokes everyone while the per-user clear leaves everyone else connected", () => {
    renderDialog();

    expect(screen.getByText(SHARED_BLAST_RADIUS).textContent).toMatch(
      /Anyone who authorized interactively loses upstream access until they authorize again/i,
    );
    // A machine-to-machine server mints from stored client credentials, so a clear only forces a
    // fresh mint. Saying it revokes access would be a lie for exactly the servers that keep working.
    expect(screen.getByText(SHARED_BLAST_RADIUS).textContent).toMatch(
      /machine-to-machine server keeps the client credentials .* mints a fresh token .* instead of losing access/i,
    );
    expect(screen.getByText(SHARED_BLAST_RADIUS).textContent).toMatch(/BYOK API keys are left alone/i);
    expect(screen.getByText(OWN_BLAST_RADIUS).textContent).toMatch(/Every other user keeps their connection/i);
  });

  it("defaults an admin to the server-level clear and calls the server-level endpoint", async () => {
    const { onCleared } = renderDialog();

    await userEvent.click(screen.getByRole("button", { name: "Disconnect" }));

    await waitFor(() => expect(deleteMCPServerOAuthToken).toHaveBeenCalledWith("sk-test", "srv-1"));
    expect(deleteMCPOAuthUserCredential).not.toHaveBeenCalled();
    expect(onCleared).toHaveBeenCalledWith("server");
  });

  it("calls the per-user revoke when the caller picks their own connection", async () => {
    const { onCleared } = renderDialog();

    await userEvent.click(screen.getByText(/Only your own connection \(affects just you\)/i));
    await userEvent.click(screen.getByRole("button", { name: "Disconnect" }));

    await waitFor(() => expect(deleteMCPOAuthUserCredential).toHaveBeenCalledWith("sk-test", "srv-1"));
    expect(deleteMCPServerOAuthToken).not.toHaveBeenCalled();
    expect(onCleared).toHaveBeenCalledWith("self");
  });

  it("defaults a non-admin to their own connection and blocks the shared clear", async () => {
    const { onCleared } = renderDialog({ isProxyAdmin: false });

    expect(screen.getByText(/Requires proxy admin/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Disconnect" }));

    await waitFor(() => expect(onCleared).toHaveBeenCalledWith("self"));
    expect(deleteMCPServerOAuthToken).not.toHaveBeenCalled();
  });

  it("labels the reauthorize entry point and still shows both blast radii", () => {
    renderDialog({ mode: "reauthorize" });

    expect(screen.getByText("Reauthorize MCP Server?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Clear & Reauthorize" })).toBeInTheDocument();
    expect(screen.getByText(SHARED_BLAST_RADIUS)).toBeInTheDocument();
    expect(screen.getByText(OWN_BLAST_RADIUS)).toBeInTheDocument();
  });
});
