{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   {{ _('Functions') }}
   --------------------

   .. autosummary::
     :nosignatures:
   {% for item in functions %}
      {{ item }}

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   {{ _('Classes') }}
   ------------------

   .. autosummary::
     :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

  {% block modules %}
  {% if modules %}
  Modules
  -------

  .. autosummary::
     :toctree:
     :recursive:
  {% for item in modules %}
     {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}
