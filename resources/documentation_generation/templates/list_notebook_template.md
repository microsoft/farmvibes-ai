# Notebooks

We present a complete list of the notebooks available in FarmVibes.AI with a short summary for each of them. Besides their description, we also include the expected disk space and running time required per notebook, considering the recommended VM size.

<br>

---------------


## Summary

We organize available notebooks in the following topics:

{% for tag_tuple, nb_list in tag_data_source -%}

<details>
<summary> {{tag_tuple[1]}} </summary>

{% for nb in nb_list %}- [`{{nb.name}}` 📓]({{nb.repo_path}})

{% endfor %}
</details>
{% endfor %}



<br>

---------------


## Notebooks description

{% for nb in data_source %}- [`{{nb.name}}` 📓]({{nb.repo_path}}) {%if nb.disk_time_req %} {{nb.disk_time_req}} {% endif %}: {{nb.description}}

{% endfor %}

