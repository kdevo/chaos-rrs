ENV_NAME := chaos

update-conda:
	conda update -n base -c defaults conda
	conda env update --file environment.yml --prune

init-conda:
	conda env create -f environment.yml

init: init-conda

update: update-conda

purge:
	#conda deactivate
	conda remove --name $(ENV_NAME) --all

test:
	py.test tests

docs:
	sphinx-build docs built-docs

chaos: ;


.PHONY: update-conda init-conda test update docs
