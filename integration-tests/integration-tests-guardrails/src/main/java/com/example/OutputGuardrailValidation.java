package com.example;

import dev.langchain4j.guardrail.OutputGuardrail;
import dev.langchain4j.guardrail.OutputGuardrailParams;
import dev.langchain4j.guardrail.OutputGuardrailResult;
import java.util.Map;

public class OutputGuardrailValidation implements OutputGuardrail {
    private static final OutputGuardrailValidation INSTANCE = new OutputGuardrailValidation();
    private OutputGuardrailParams params;

    private OutputGuardrailValidation() {}

    public static OutputGuardrailValidation getInstance() {
        return INSTANCE;
    }

    public OutputGuardrailResult validate(OutputGuardrailParams params) {
        this.params = params;
        return success();
    }

    public void reset() {
        this.params = null;
    }

    public String spyUserMessageTemplate() {
        return params.commonParams().userMessageTemplate();
    }

    public Map<String, Object> spyVariables() {
        return params.commonParams().variables();
    }
}
