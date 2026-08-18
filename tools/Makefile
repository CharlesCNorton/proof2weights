# proof2weights Makefile
#
# Builds the verified Coq -> safetensors pipeline.
# Run from tools/ directory or use: make -C tools
#
# Requirements: coqc (or rocq compile), ocamlc, python with safetensors

COQC ?= coqc
OCAMLC ?= ocamlc
PYTHON ?= python

THEORIES = ../theories
STEM = phases1_15_complete
MODULE = Phases1_15_complete

.PHONY: all clean coq ocaml run verify

all: coq ocaml run

# Compile Coq. This also emits the OCaml extraction next to the source.
coq: $(THEORIES)/$(MODULE).vo

$(THEORIES)/$(MODULE).vo $(THEORIES)/$(STEM).ml $(THEORIES)/$(STEM).mli: $(THEORIES)/$(MODULE).v
	cd $(THEORIES) && $(COQC) -R . "" $(MODULE).v

# Compile OCaml (extraction + I/O wrapper)
ocaml: proof2weights.exe

proof2weights.exe: $(THEORIES)/$(STEM).mli $(THEORIES)/$(STEM).ml io.ml main.ml
	$(OCAMLC) -I $(THEORIES) $^ -o $@

# Run and export networks
run: proof2weights.exe
	./proof2weights.exe

majority.safetensors mod3.safetensors: run

# Verify with Python
verify: majority.safetensors mod3.safetensors
	$(PYTHON) -c "from safetensors import safe_open; \
	f = safe_open('majority.safetensors', framework='numpy'); \
	w = f.get_tensor('layer1.weight'); \
	b = f.get_tensor('layer1.bias'); \
	assert list(w.flatten()) == [1,1,1,1,1,1,1,1], 'weight mismatch'; \
	assert list(b.flatten()) == [-5], 'bias mismatch'; \
	print('majority: OK')"
	$(PYTHON) -c "from safetensors import safe_open; \
	f = safe_open('mod3.safetensors', framework='numpy'); \
	assert len(list(f.keys())) == 6, 'expected 6 tensors'; \
	print('mod3: OK')"

clean:
	rm -f *.cmo *.cmi *.exe *.safetensors
	cd $(THEORIES) && rm -f *.cmo *.cmi *.vo *.vok *.vos *.glob
