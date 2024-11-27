# {{data_source.name}}

{{data_source.description.short_description}} {{data_source.description.long_description}}

```{mermaid}
    {{data_source.mermaid_diagram}}
```

## Sources

{% for source_name, source_desc in data_source.description.inputs.items() -%}

- **{{source_name}}**: {{source_desc}}

{% endfor -%}

## Sinks

{% for sink_name, sink_desc in data_source.description.outputs.items() -%}

- **{{sink_name}}**: {{sink_desc}}

{% endfor -%}

{% if data_source.description.parameters -%}
## Parameters

{% for param_name, param_desc in data_source.description.parameters.items() -%}

- **{{param_name}}**: {% if param_desc is string %}{{param_desc}}{% else %}{{param_desc[0]}}{% endif %}

{% endfor -%}
{% endif -%}

{% if data_source.description.task_descriptions -%}
## Tasks

{% for task_name, task_desc in data_source.description.task_descriptions.items() -%}

- **{{task_name}}**: {{task_desc}}

{% endfor -%}
{% endif -%}

## Workflow Yaml

```yaml

{{data_source.yaml}}

```
