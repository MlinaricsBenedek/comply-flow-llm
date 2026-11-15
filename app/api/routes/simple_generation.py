from fastapi import APIRouter, Request
from pydantic import Field, field_validator, model_validator

from app.api.errors import APIError, APIErrorResponse
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.llm_service import InvalidModelParameters, ModelExecutionError, UnknownModelError


class SimpleGenerationRequest(GenerationRequest):
    rule_decision: str = Field(min_length=1)
    rule_reason: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def extract_rule_data(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        input_data = data.get("input_data")
        if not isinstance(input_data, dict):
            return data
        rule_result = input_data.get("rule_result", input_data.get("ruleResult"))
        sources = [input_data]
        if isinstance(rule_result, dict):
            sources.append(rule_result)
        aliases = {
            "rule_decision": ("rule_decision", "ruleDecision", "decision"),
            "rule_reason": ("rule_reason", "ruleReason", "reason"),
            "matched_rules": ("matched_rules", "matchedRules"),
        }
        for field, names in aliases.items():
            if field in data and data[field] is not None:
                continue
            data.pop(field, None)
            for source in sources:
                for name in names:
                    if name in source:
                        data[field] = source[name]
                        break
                if field in data:
                    break
        return data

    @field_validator("rule_decision", "rule_reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be blank.")
        return value

    @field_validator("matched_rules")
    @classmethod
    def nonblank_rules(cls, value: list[str]) -> list[str]:
        if any(not rule.strip() for rule in value):
            raise ValueError("Rules must not be blank.")
        return value


router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerationResponse,
    responses={
        status: {"model": APIErrorResponse}
        for status in (400, 422, 500, 503, 504)
    },
)
def generate_response(payload: SimpleGenerationRequest, request: Request) -> GenerationResponse:
    service = request.app.state.generation_service
    if service is None:
        raise APIError(503, "GENERATION_UNAVAILABLE", "The generation service is unavailable.")
    try:
        result = service.generate_result(
            rule_decision=payload.rule_decision,
            rule_reason=payload.rule_reason,
            matched_rules=payload.matched_rules,
            model_name=payload.model_name,
            model_parameters=(
                payload.model_parameters.model_dump(exclude_none=True)
                if payload.model_parameters is not None
                else None
            ),
        )
    except TimeoutError as error:
        raise APIError(504, "GENERATION_TIMEOUT", "The generation request timed out.") from error
    except UnknownModelError as error:
        raise APIError(400, "MODEL_NOT_SUPPORTED", "The requested model is not supported.") from error
    except InvalidModelParameters as error:
        raise APIError(400, "MODEL_PARAMETERS_INVALID", "The model parameters are invalid.") from error
    except (KeyError, ValueError) as error:
        raise APIError(503, "GENERATION_UNAVAILABLE", "The generation prompt is invalid.") from error
    except ModelExecutionError as error:
        request.state.llm_model_name = payload.model_name or service.llm_service.default_model
        raise APIError(503, "LLM_UNAVAILABLE", "The language model is unavailable.") from error

    return GenerationResponse(
        response=result.response,
        prompt_version=result.prompt_version,
        model_name=result.model_name,
        model_id=result.model_id,
        model_parameters=result.model_parameters,
    )
