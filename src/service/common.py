from src.service.model import WarningResponse

def get_warning_response(warnings: list[str] | None) -> WarningResponse | None:
    if not warnings:
        return None
    return WarningResponse(
        num_warnings=len(warnings),
        last_warning=warnings[-1],
    )