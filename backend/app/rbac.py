from enum import Enum

from fastapi import Depends, Header, HTTPException


class Role(str, Enum):
    ADMINISTRATOR = "Administrator"
    ANALYST = "Analyst"
    REVIEWER = "Reviewer"
    AUDITOR = "Auditor"
    SYSTEM = "SYSTEM"


ROLE_PERMISSIONS = {
    Role.ADMINISTRATOR: {
        "screen": True,
        "review": True,
        "audit": True,
        "admin": True,
    },

    Role.ANALYST: {
        "screen": True,
        "review": False,
        "audit": True,
        "admin": False,
    },

    Role.REVIEWER: {
        "screen": False,
        "review": True,
        "audit": False,
        "admin": False,
    },

    Role.AUDITOR: {
        "screen": False,
        "review": False,
        "audit": True,
        "admin": False,
    },

    Role.SYSTEM: {
        "screen": True,
        "review": False,
        "audit": True,
        "admin": False,
    },
}


def get_current_role(
    x_shieldx_role: str = Header(...),
) -> Role:
    """
    Resolve the ShieldX role supplied by the authenticated layer.

    This header-based mechanism is for Phase 1B authorization testing.
    It is NOT the final production authentication mechanism.
    """

    try:
        return Role(x_shieldx_role)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Invalid ShieldX role.",
        )


def require_permission(permission: str):
    """
    Create a FastAPI dependency that requires a specific permission.
    """

    def dependency(
        role: Role = Depends(get_current_role),
    ):
        permissions = ROLE_PERMISSIONS.get(role, {})

        if not permissions.get(permission, False):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Role '{role.value}' is not authorized "
                    f"for '{permission}' operations."
                ),
            )

        return role

    return dependency