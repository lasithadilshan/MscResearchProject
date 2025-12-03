"""
Optimized Cucumber Script Generation Module for Streamlit Integration
Converts test cases to Cucumber BDD scripts with improved performance and maintainability
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import streamlit as st


@dataclass
class CucumberConfig:
    """Configuration for Cucumber script generation"""
    framework: str = 'selenium'
    language: str = 'java'
    pattern: str = 'page_object'
    tags: List[str] = field(default_factory=lambda: ['@automated'])
    include_hooks: bool = True
    include_examples: bool = True
    include_negative: bool = True
    show_confidence: bool = True
    
    # Framework mappings
    FRAMEWORKS = {
        'selenium': 'Selenium WebDriver for web automation',
        'appium': 'Appium for mobile app testing',
        'restassured': 'RestAssured for API testing',
        'playwright': 'Playwright for modern web testing'
    }
    
    PATTERNS = {
        'page_object': 'Page Object Model pattern with separate page classes',
        'screenplay': 'Screenplay pattern with actors, tasks, and questions',
        'traditional': 'Traditional procedural step definitions'
    }
    
    def get_framework_description(self) -> str:
        return self.FRAMEWORKS.get(self.framework, self.framework)
    
    def get_pattern_description(self) -> str:
        return self.PATTERNS.get(self.pattern, self.pattern)


class CucumberPromptBuilder:
    """Builder class for constructing optimized Cucumber prompts"""
    
    @staticmethod
    def build_basic_prompt(test_case_text: str) -> str:
        """Build basic prompt for standard Cucumber generation"""
        return f"""You are an expert test automation engineer specializing in BDD and Cucumber frameworks.
Transform the following test cases into a complete, production-ready Cucumber test suite.

## INPUT TEST CASES:
{test_case_text}

## REQUIREMENTS:

### 1. FEATURE FILE (.feature)
- Feature name and description explaining business value
- Background section for common preconditions (if applicable)
- Scenarios covering all test cases with proper tags (@smoke, @regression, @critical)
- Scenario Outlines with Examples for data-driven tests
- Clear Given-When-Then pattern with "And"/"But" for readability

### 2. STEP DEFINITIONS (Java)
- Package declaration and necessary imports (Selenium WebDriver, Cucumber, assertions)
- @Given, @When, @Then, @And annotations with regex patterns
- Page Object Model references
- Explicit waits and element locators
- Error handling, logging, and parameterized steps
- Setup/teardown hooks (@Before, @After)
- Data table handling for complex inputs

### 3. BEST PRACTICES:
- Atomic and reusable steps expressing intent, not implementation
- Avoid technical jargon in Gherkin
- Include positive and negative scenarios
- Follow DRY principle with meaningful names
- Include proper assertions

### 4. OUTPUT FORMAT:

**FEATURE FILE (feature_name.feature):**
```gherkin
[Complete feature file]
```

**STEP DEFINITIONS (StepDefinitions.java):**
```java
[Complete Java step definitions]
```

**TEST DATA NOTES:**
[Test data management recommendations]

**EXECUTION NOTES:**
[How to run tests and dependencies]

### 5. ADDITIONAL CONSIDERATIONS:
- Infer reasonable locator strategies for UI elements
- Include wait strategies for dynamic elements
- Consider cross-browser compatibility
- Add data validation and cleanup steps

Transform the test cases into a professional Cucumber test suite."""

    @staticmethod
    def build_advanced_prompt(test_case_text: str, config: CucumberConfig) -> str:
        """Build advanced prompt with custom configuration"""
        
        # Build conditional sections
        negative_scenarios = "- Include negative test scenarios" if config.include_negative else ""
        hooks_section = "- @Before and @After hooks for setup/teardown" if config.include_hooks else ""
        
        pattern_section = ""
        if config.pattern == 'page_object':
            pattern_section = f"""
**PAGE OBJECTS:**
```{config.language}
[Page object implementation]
```"""
        elif config.pattern == 'screenplay':
            pattern_section = f"""
**SCREENPLAY COMPONENTS:**
```{config.language}
[Actor, Task, and Question classes]
```"""
        
        return f"""You are an expert test automation engineer specializing in BDD and Cucumber frameworks.
Transform the following test cases into a complete, production-ready Cucumber test suite.

## INPUT TEST CASES:
{test_case_text}

## CONFIGURATION:
- **Framework**: {config.get_framework_description()}
- **Language**: {config.language.upper()}
- **Design Pattern**: {config.get_pattern_description()}
- **Tags**: {', '.join(config.tags)}
- **Hooks**: {'Included' if config.include_hooks else 'Excluded'}
- **Example Tables**: {'Included' if config.include_examples else 'Excluded'}
- **Negative Scenarios**: {'Included' if config.include_negative else 'Excluded'}

## DELIVERABLES:

### 1. FEATURE FILE
- Clear feature description with business context
- Well-structured scenarios using Given-When-Then
- Scenario Outlines with Examples tables (if enabled)
- Appropriate tags for test categorization
- Background section for common steps
{negative_scenarios}

### 2. STEP DEFINITIONS ({config.language.upper()})
- Complete import statements for {config.framework}
- {config.pattern} implementation
- Parameterized steps with regex patterns
- Proper error handling and assertions
- Wait strategies for dynamic elements
{hooks_section}

### 3. SUPPORTING CODE:
- {'Page Object classes with locators and methods' if config.pattern == 'page_object' else 'Supporting classes'}
- Utility methods for common operations
- Configuration management approach

### 4. BEST PRACTICES:
- Declarative steps (not imperative)
- Single responsibility principle
- Explicit waits over implicit waits
- Meaningful assertions
- Separate test data from logic
- Ensure test independence

## OUTPUT FORMAT:

**FEATURE FILE:**
```gherkin
[Complete feature file]
```

**STEP DEFINITIONS:**
```{config.language}
[Complete step definitions]
```
{pattern_section}

**EXECUTION INSTRUCTIONS:**
[How to run these tests]

Transform the test cases following all specifications above."""


class CucumberScriptGenerator:
    """Main class for generating Cucumber scripts with Streamlit integration"""
    
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.prompt_builder = CucumberPromptBuilder()
    
    def _validate_inputs(self, test_case_text: str) -> bool:
        """Validate input parameters"""
        if not self.qa_chain:
            st.error("❌ QA chain not initialized. Please upload a BRD document first.")
            return False
        
        if not test_case_text or not test_case_text.strip():
            st.warning("❌ Please provide test case text.")
            return False
        
        return True
    
    def _generate_script(self, prompt: str) -> tuple[str, float]:
        """Generate script and return result with execution time"""
        start_time = time.time()
        response = self.qa_chain.invoke({"query": prompt})
        end_time = time.time()
        
        return response['result'], end_time - start_time
    
    def _display_results_streamlit(self, result: str, execution_time: float, 
                                   show_confidence: bool = False, 
                                   test_case_text: str = None,
                                   prompt: str = None,
                                   calculate_confidence_fn=None,
                                   calculate_match_fn=None,
                                   display_metrics_fn=None):
        """Display generation results in Streamlit with optional confidence metrics"""
        st.subheader("Generated Cucumber Script")
        st.markdown(result)
        
        if show_confidence and test_case_text and prompt and all([calculate_confidence_fn, calculate_match_fn, display_metrics_fn]):
            st.subheader("Quality Assessment")
            
            confidence_score = calculate_confidence_fn(prompt, result)
            match_score = calculate_match_fn(result, test_case_text)
            display_metrics_fn(confidence_score, match_score)
        
        st.info(f"⏱️ Generation time: {execution_time:.2f} seconds")
    
    def generate_basic(self, test_case_text: str, show_confidence: bool = True,
                      calculate_confidence_fn=None, calculate_match_fn=None, 
                      display_metrics_fn=None) -> Optional[str]:
        """Generate basic Cucumber script with default settings"""
        if not self._validate_inputs(test_case_text):
            return None
        
        with st.spinner("🥒 Generating Cucumber Script..."):
            prompt = self.prompt_builder.build_basic_prompt(test_case_text)
            result, execution_time = self._generate_script(prompt)
        
        self._display_results_streamlit(
            result, 
            execution_time, 
            show_confidence, 
            test_case_text, 
            prompt,
            calculate_confidence_fn,
            calculate_match_fn,
            display_metrics_fn
        )
        
        return result
    
    def generate_advanced(self, test_case_text: str, 
                         config: Optional[CucumberConfig] = None,
                         calculate_confidence_fn=None, 
                         calculate_match_fn=None, 
                         display_metrics_fn=None) -> Optional[str]:
        """Generate advanced Cucumber script with custom configuration"""
        if not self._validate_inputs(test_case_text):
            return None
        
        config = config or CucumberConfig()
        
        with st.spinner(f"🥒 Generating Cucumber Script with {config.framework} framework..."):
            prompt = self.prompt_builder.build_advanced_prompt(test_case_text, config)
            result, execution_time = self._generate_script(prompt)
        
        st.success(f"✅ Generated Cucumber Script ({config.framework}, {config.pattern})")
        
        self._display_results_streamlit(
            result, 
            execution_time, 
            config.show_confidence, 
            test_case_text, 
            prompt,
            calculate_confidence_fn,
            calculate_match_fn,
            display_metrics_fn
        )
        
        return result


# Convenience functions for Streamlit integration
def generate_cucumber_script_streamlit(qa_chain, test_case_text: str, 
                                      show_confidence: bool = True,
                                      calculate_confidence_fn=None,
                                      calculate_match_fn=None,
                                      display_metrics_fn=None) -> Optional[str]:
    """Generate Cucumber script with basic configuration (Streamlit compatible)"""
    generator = CucumberScriptGenerator(qa_chain)
    return generator.generate_basic(
        test_case_text, 
        show_confidence,
        calculate_confidence_fn,
        calculate_match_fn,
        display_metrics_fn
    )


def generate_cucumber_script_advanced_streamlit(qa_chain, test_case_text: str, 
                                               config: Optional[Dict[str, Any]] = None,
                                               calculate_confidence_fn=None,
                                               calculate_match_fn=None,
                                               display_metrics_fn=None) -> Optional[str]:
    """Generate Cucumber script with advanced configuration (Streamlit compatible)"""
    generator = CucumberScriptGenerator(qa_chain)
    
    # Convert dict config to CucumberConfig object
    if config and isinstance(config, dict):
        config = CucumberConfig(**config)
    
    return generator.generate_advanced(
        test_case_text, 
        config,
        calculate_confidence_fn,
        calculate_match_fn,
        display_metrics_fn
    )