from enum import Enum

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .core.security import decode_access_token

security = HTTPBearer()


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
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Role:
    """
    Resolve the ShieldX role from the authenticated JWT.
    """

    try:
        payload = decode_access_token(credentials.credentials)

        role_value = payload.get("role")

        if not role_value:
            raise HTTPException(
                status_code=401,
                detail="Authentication token has no role.",
            )

        return Role(role_value)

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
        )
def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> int:
    """
    Resolve the authenticated ShieldX user's database ID from the JWT.
    """

    try:
        payload = decode_access_token(
            credentials.credentials
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication token has no user ID.",
            )

        return int(user_id)

    except HTTPException:
        raise

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Authentication token has an invalid user ID.",
        )

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
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