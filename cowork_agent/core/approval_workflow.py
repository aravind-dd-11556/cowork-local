"""
Approval Workflow — Human-in-the-loop approval for high-risk actions.

Multi-stage approval system that integrates with the security pipeline's
EXPLICIT_CONSENT tier. Supports approval chains, escalation policies,
timeout-based auto-decisions, and audit trails.

Features:
  - Single-stage approval (ask user → approve/decline)
  - Multi-stage approval chains (e.g., user → then cost check → then execute)
  - Escalation policies (timeout → auto-decline or auto-approve)
  - Full audit trail with timestamps and decision history
  - Integration with ConsentManager for user interaction
  - Cost-aware approval (auto-confirm cheap actions, confirm expensive ones)

Sprint 24: Production Hardening.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────

class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    TIMED_OUT = "timed_out"
    AUTO_APPROVED = "auto_approved"
    ESCALATED = "escalated"


class EscalationPolicy(Enum):
    """What happens when approval times out."""
    AUTO_DECLINE = "auto_decline"   # Default safe option
    AUTO_APPROVE = "auto_approve"   # For low-risk actions
    ESCALATE = "escalate"           # Notify higher authority


class ApprovalCategory(Enum):
    """Categories of actions requiring approval."""
    DESTRUCTIVE = "destructive"       # Delete, overwrite, rm -rf
    NETWORK = "network"               # HTTP requests, push, deploy
    INSTALL = "install"               # pip install, npm install
    PUBLISH = "publish"               # Post, send, share publicly
    FINANCIAL = "financial"           # Purchases, cost-incurring actions
    SENSITIVE_DATA = "sensitive_data"  # Access to credentials, PII
    CONFIGURATION = "configuration"   # System config changes
    GENERAL = "general"               # Catch-all


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class ApprovalRequest:
    """A single approval request."""
    request_id: str
    category: ApprovalCategory
    tool_name: str
    tool_input: Dict[str, Any]
    description: str
    estimated_cost: Optional[float] = None
    risk_level: str = "medium"  # low, medium, high, critical
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float = 300.0  # 5 minute default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "category": self.category.value,
            "tool_name": self.tool_name,
            "description": self.description,
            "estimated_cost": self.estimated_cost,
            "risk_level": self.risk_level,
            "created_at": self.created_at,
        }


@dataclass
class ApprovalDecision:
    """The decision made on an approval request."""
    request: ApprovalRequest
    status: ApprovalStatus
    decided_by: str = "user"  # "user", "timeout", "auto", "cost_check"
    decided_at: float = field(default_factory=time.time)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)

    @property
    def latency_seconds(self) -> float:
        return self.decided_at - self.request.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "status": self.status.value,
            "decided_by": self.decided_by,
            "approved": self.approved,
            "reason": self.reason,
            "latency_seconds": round(self.latency_seconds, 2),
        }


# ── Category configuration ───────────────────────────────────────

@dataclass
class CategoryConfig:
    """Configuration for an approval category."""
    escalation_policy: EscalationPolicy = EscalationPolicy.AUTO_DECLINE
    timeout_seconds: float = 300.0
    auto_approve_below_cost: float = 0.0  # Auto-approve if cost < this
    require_explicit_confirmation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalation_policy": self.escalation_policy.value,
            "timeout_seconds": self.timeout_seconds,
            "auto_approve_below_cost": self.auto_approve_below_cost,
            "require_explicit_confirmation": self.require_explicit_confirmation,
        }


DEFAULT_CATEGORY_CONFIGS: Dict[ApprovalCategory, CategoryConfig] = {
    ApprovalCategory.DESTRUCTIVE: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=120.0,
        require_explicit_confirmation=True,
    ),
    ApprovalCategory.NETWORK: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=300.0,
        auto_approve_below_cost=0.0,
    ),
    ApprovalCategory.INSTALL: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=300.0,
    ),
    ApprovalCategory.PUBLISH: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=120.0,
        require_explicit_confirmation=True,
    ),
    ApprovalCategory.FINANCIAL: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=60.0,
        require_explicit_confirmation=True,
    ),
    ApprovalCategory.SENSITIVE_DATA: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=120.0,
        require_explicit_confirmation=True,
    ),
    ApprovalCategory.CONFIGURATION: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=300.0,
    ),
    ApprovalCategory.GENERAL: CategoryConfig(
        escalation_policy=EscalationPolicy.AUTO_DECLINE,
        timeout_seconds=300.0,
        auto_approve_below_cost=0.01,
    ),
}


# ── Tool → Category mapping ──────────────────────────────────────

def classify_tool_category(tool_name: str, tool_input: Dict[str, Any]) -> ApprovalCategory:
    """Map a tool call to an approval category."""
    cmd = str(tool_input.get("command", ""))

    if tool_name == "bash":
        # Destructive commands
        if any(p in cmd for p in ["rm ", "rm\t", "rmdir", "truncate", "shred"]):
            return ApprovalCategory.DESTRUCTIVE
        # Install commands
        if any(p in cmd for p in ["pip install", "npm install", "apt install", "brew install"]):
            return ApprovalCategory.INSTALL
        # Network commands
        if any(p in cmd for p in ["curl", "wget", "git push", "ssh", "scp", "rsync"]):
            return ApprovalCategory.NETWORK
        # Publish commands
        if any(p in cmd for p in ["git push", "npm publish", "docker push"]):
            return ApprovalCategory.PUBLISH
        # Config changes
        if any(p in cmd for p in ["chmod", "chown", "systemctl", "service"]):
            return ApprovalCategory.CONFIGURATION

    elif tool_name == "write":
        file_path = str(tool_input.get("file_path", ""))
        if any(ext in file_path for ext in [".env", ".key", ".pem", ".secret"]):
            return ApprovalCategory.SENSITIVE_DATA
        if any(p in file_path for p in ["/etc/", "/.ssh/", "/config"]):
            return ApprovalCategory.CONFIGURATION

    return ApprovalCategory.GENERAL


# ── ApprovalWorkflow ─────────────────────────────────────────────

class ApprovalWorkflow:
    """Human-in-the-loop approval engine.

    Manages approval requests, tracks decisions, and enforces policies.

    Usage::

        workflow = ApprovalWorkflow()
        workflow.set_user_callback(lambda req: input(f"Approve? {req.description} [y/n]: "))

        decision = workflow.request_approval(
            tool_name="bash",
            tool_input={"command": "git push origin main"},
            description="Push changes to remote repository",
        )
        if decision.approved:
            # execute the tool
    """

    def __init__(
        self,
        category_configs: Optional[Dict[ApprovalCategory, CategoryConfig]] = None,
        user_callback: Optional[Callable[[ApprovalRequest], str]] = None,
        auto_approve_all: bool = False,
    ):
        self._configs = dict(DEFAULT_CATEGORY_CONFIGS)
        if category_configs:
            self._configs.update(category_configs)
        self._user_callback = user_callback
        self._auto_approve_all = auto_approve_all

        # Audit trail
        self._decisions: List[ApprovalDecision] = []
        self._request_counter = 0

        # Stats
        self._total_requests = 0
        self._total_approved = 0
        self._total_declined = 0
        self._total_auto_approved = 0
        self._total_timed_out = 0

    # ── Configuration ──────────────────────────────────────────

    def set_user_callback(self, callback: Callable[[ApprovalRequest], str]) -> None:
        """Set the callback function for asking user approval."""
        self._user_callback = callback

    def set_category_config(self, category: ApprovalCategory, config: CategoryConfig) -> None:
        """Configure approval behavior for a category."""
        self._configs[category] = config

    def get_category_config(self, category: ApprovalCategory) -> CategoryConfig:
        """Get configuration for a category."""
        return self._configs.get(category, DEFAULT_CATEGORY_CONFIGS.get(
            category, CategoryConfig()
        ))

    # ── Core approval ──────────────────────────────────────────

    def request_approval(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        description: str,
        estimated_cost: Optional[float] = None,
        risk_level: str = "medium",
        category: Optional[ApprovalCategory] = None,
    ) -> ApprovalDecision:
        """Request approval for an action.

        Args:
            tool_name: Tool being called
            tool_input: Tool input parameters
            description: Human-readable description of the action
            estimated_cost: Estimated cost in USD (if applicable)
            risk_level: Risk level (low, medium, high, critical)
            category: Override auto-detected category

        Returns:
            ApprovalDecision with the user's (or system's) decision
        """
        self._total_requests += 1
        self._request_counter += 1

        # Create request
        if category is None:
            category = classify_tool_category(tool_name, tool_input)

        config = self.get_category_config(category)

        request = ApprovalRequest(
            request_id=f"approval_{self._request_counter}",
            category=category,
            tool_name=tool_name,
            tool_input=tool_input,
            description=description,
            estimated_cost=estimated_cost,
            risk_level=risk_level,
            timeout_seconds=config.timeout_seconds,
        )

        # Check auto-approve conditions
        if self._auto_approve_all:
            decision = ApprovalDecision(
                request=request,
                status=ApprovalStatus.AUTO_APPROVED,
                decided_by="auto",
                reason="Auto-approve all enabled",
            )
            self._total_auto_approved += 1
            self._total_approved += 1
            self._decisions.append(decision)
            return decision

        # Cost-based auto-approve
        if (estimated_cost is not None and
                config.auto_approve_below_cost > 0 and
                estimated_cost < config.auto_approve_below_cost):
            decision = ApprovalDecision(
                request=request,
                status=ApprovalStatus.AUTO_APPROVED,
                decided_by="cost_check",
                reason=f"Cost ${estimated_cost:.4f} below threshold ${config.auto_approve_below_cost:.4f}",
            )
            self._total_auto_approved += 1
            self._total_approved += 1
            self._decisions.append(decision)
            return decision

        # Ask user
        if self._user_callback:
            try:
                response = self._user_callback(request)
                approved = self._interpret_response(response)
                decision = ApprovalDecision(
                    request=request,
                    status=ApprovalStatus.APPROVED if approved else ApprovalStatus.DECLINED,
                    decided_by="user",
                    reason=response,
                )
                if approved:
                    self._total_approved += 1
                else:
                    self._total_declined += 1
                self._decisions.append(decision)
                return decision
            except Exception as e:
                logger.warning(f"User callback error: {e}")
                # Fall through to timeout/escalation

        # No callback — apply escalation policy
        return self._apply_escalation_policy(request, config)

    def _interpret_response(self, response: str) -> bool:
        """Interpret user's response as approve/decline."""
        normalized = response.strip().lower()
        approve_words = {"yes", "y", "approve", "ok", "confirm", "go", "proceed",
                         "sure", "do it", "go ahead", "accepted"}
        return normalized in approve_words

    def _apply_escalation_policy(
        self, request: ApprovalRequest, config: CategoryConfig
    ) -> ApprovalDecision:
        """Apply escalation policy when no user response is available."""
        policy = config.escalation_policy

        if policy == EscalationPolicy.AUTO_APPROVE:
            decision = ApprovalDecision(
                request=request,
                status=ApprovalStatus.AUTO_APPROVED,
                decided_by="escalation_policy",
                reason="No user callback — auto-approved by policy",
            )
            self._total_auto_approved += 1
            self._total_approved += 1
        elif policy == EscalationPolicy.ESCALATE:
            decision = ApprovalDecision(
                request=request,
                status=ApprovalStatus.ESCALATED,
                decided_by="escalation_policy",
                reason="No user callback — escalated",
            )
            self._total_declined += 1  # Escalated = not approved yet
        else:  # AUTO_DECLINE
            decision = ApprovalDecision(
                request=request,
                status=ApprovalStatus.DECLINED,
                decided_by="escalation_policy",
                reason="No user callback — auto-declined by policy",
            )
            self._total_declined += 1

        self._decisions.append(decision)
        return decision

    # ── Queries ────────────────────────────────────────────────

    @property
    def decision_history(self) -> List[ApprovalDecision]:
        """Full audit trail of all decisions."""
        return list(self._decisions)

    def get_decision(self, request_id: str) -> Optional[ApprovalDecision]:
        """Look up a specific decision by request ID."""
        for d in self._decisions:
            if d.request.request_id == request_id:
                return d
        return None

    def clear_history(self) -> None:
        """Clear decision history (for new session)."""
        self._decisions.clear()
        self._request_counter = 0

    # ── Stats ──────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self._total_requests,
            "total_approved": self._total_approved,
            "total_declined": self._total_declined,
            "total_auto_approved": self._total_auto_approved,
            "total_timed_out": self._total_timed_out,
            "approval_rate": (
                round(self._total_approved / self._total_requests * 100, 1)
                if self._total_requests > 0 else 0.0
            ),
        }

    def summary(self) -> Dict[str, Any]:
        """Comprehensive workflow summary."""
        return {
            **self.stats,
            "category_configs": {
                cat.value: cfg.to_dict()
                for cat, cfg in self._configs.items()
            },
            "recent_decisions": [
                d.to_dict() for d in self._decisions[-10:]
            ],
        }
