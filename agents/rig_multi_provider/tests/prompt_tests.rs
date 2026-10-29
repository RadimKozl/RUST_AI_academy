use rig_multi_provider::PromptTemplate;

#[test]
fn test_prompt_template_single_variable() {
    let template = PromptTemplate::new("What is a good name for a company that makes {product}?");
    let result = template.format_product("colorful socks");

    assert_eq!(
        result,
        "What is a good name for a company that makes colorful socks?"
    );
}

#[test]
fn test_prompt_template_multiple_variables() {
    let template = PromptTemplate::new("Hello {name}, welcome to our company making {product}!");
    let result = template.format_name_and_product("Lukas", "colorful socks");

    assert_eq!(
        result,
        "Hello Lukas, welcome to our company making colorful socks!"
    );
}