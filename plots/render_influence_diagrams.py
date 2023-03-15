"""Render influence diagrams for example decision problems."""

import markdown

text = """
# Influence Diagrams for Example Decision Problems

## Ventilation Scheduling for Indoor Air Quailty

~~~mermaid
flowchart LR
    d1 --> c1{Ventilation\\n energy cost}
    d1[Schedule ventilation\\n rate] --> o1(Risk of\\n infection)

    d3[Building monitoring\\n system] --> o2

    o1 --> c2{Staff illness\\n cost}

    o2(Building occupancy) --> o1

    d3 --> c3{Smart control\\n system cost}
~~~


## Air Source Heatpump Maintenance Scheduling

~~~mermaid
flowchart LR
    d1[Maintenance\\n frequency] --> c2{Maintenance\\n cost}
    d1 --> o2(("#946;"))

    o1 --> c1{Electricity\\n cost}

    o2((&#946)) --> o1((SPF))
    o0((Heating\\n load)) --> o1((SPF))
    o3((&#945)) --> o1((SPF))

    d2 --> o3
    d2[Smart\\n meter] --> c3{Meter\\n cost}

    o0 --> o3
~~~

## Ground Source Heatpump System Design

~~~mermaid
flowchart LR
    d1[Design borehole\\n length] --> c1{Drilling\\n cost}
    d1 --> o1(System energy\\n usage)
    o1 --> c2{System\\n operation\\n cost}

    d2[Measure ground\\n thermal cond.] --> o2(Ground thermal\\n conductivity, &#955<SUB>ground</SUB>)
    d2 --> c3{Ground\\n surveying\\n cost}
    o2 --> o1
~~~

"""

html = markdown.markdown(text, extensions=['md_mermaid'])

html = '<script src="https://unpkg.com/mermaid@8.14.0/dist/mermaid.min.js"></script>\n' + html

with open("influence_diagrams.html", "w", errors="xmlcharrefreplace") as output_file:
    output_file.write(html)