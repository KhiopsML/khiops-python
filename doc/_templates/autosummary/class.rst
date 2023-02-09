{{ name | escape | underline}}
Class in module ``{{ module }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

  {% block methods %}
    {% if methods %}
      .. rubric:: {{ _('Methods') }}
      .. autosummary::
        :nosignatures:
        {% for item in methods %}
          {% if item != "__init__" %}
            ~{{ item }}
          {% endif %}
        {%- endfor %}
   {% endif %}
 {% endblock %}

  {% block attributes %}
    {% if attributes %}
     .. rubric:: {{ _('Attributes') }}
     .. autosummary::
       {% for item in attributes %}
          ~{{ name }}.{{ item }}
       {%- endfor %}
    {% endif %}
  {% endblock %}
