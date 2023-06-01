import correctionlib.schemav2 as cs

btag_var_category = cs.Category(
    nodetype="category",
    input="direction",
    content=[
        cs.CategoryItem(
            key="up",
            value=cs.Formula(
                nodetype="formula",
                parser="TFormula",
                variables=["pt"],
                expression="1.0 + (x*0.075 / 50)"
            )
        ),
        cs.CategoryItem(
            key="down",
            value=cs.Formula(
                nodetype="formula",
                parser="TFormula",
                variables=["pt"],
                expression="1.0 - (x*0.075 / 50)"
            )
        ),
    ],
    default=1.0
)

evt_systs = cs.Correction(
    name="event_systematics",
    description="Calculates a multiplicative event weight for the selected systematic variation",
    version=1,
    inputs=[
        cs.Variable(name="syst_name", type="string", description="Systematic name"),
        cs.Variable(name="direction", type="string", description="Variation direction"),
        cs.Variable(name="pt", type="real", description="One specific object pt from each event")
    ],
    output=cs.Variable(name="weight", type="real", description="Multiplicative event weight"),
    data=cs.Category(
        nodetype="category",
        input="syst_name",
        content=[
            cs.CategoryItem(
                key="scale_var",
                value=cs.Category(
                    nodetype="category",
                    input="direction",
                    content=[
                        cs.CategoryItem(
                            key="up",
                            value=1.0 + 0.025
                        ),
                        cs.CategoryItem(
                            key="down",
                            value=1.0 - 0.025
                        ),
                    ],
                    default=1.0
                )
            ),

            ############################################
            # Very compact, but requires proper handling in the processor
            cs.CategoryItem(
                key="btag_var",
                value=btag_var_category
            ),
            ### OR ###
            # Super redundant info, as the only difference is which jet pt is used
            cs.CategoryItem(
                key="btag_var_1",
                value=btag_var_category
            ),
            cs.CategoryItem(
                key="btag_var_2",
                value=btag_var_category
            ),
            cs.CategoryItem(
                key="btag_var_3",
                value=btag_var_category
            ),
            cs.CategoryItem(
                key="btag_var_4",
                value=btag_var_category
            ),
            ############################################
        ],
        default=1.0
    ),
)

cset = cs.CorrectionSet(
    schema_version=2,
    corrections=[
        evt_systs
    ]
)

with open("corrections.json", "w") as fout:
    fout.write(cset.json(exclude_unset=True))