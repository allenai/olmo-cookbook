# olmo-cookbook
OLMost every training recipe you need to perform data interventions with the OLMo family of models.

### Setup
For now, here is the install. I (@davidheineman) will eventually fix the dependency conflicts with `OLMo-ladder` once the consistent rankings code is merged in that repo.

```sh
# Install cookbook
pip install -e ".[all]"
pip install -e ".[eval]"

# Install model ladder dependency
git clone https://github.com/allenai/OLMo-ladder
cd OLMo-ladder
git checkout datados
pip install -e ".[all]" # "ladder @ git+https://github.com/allenai/OLMo-ladder.git@datados"

# Demo analysis notebook
python src/cookbook/eval/scripts/analysis/demo.py # will save outputs in img/
```