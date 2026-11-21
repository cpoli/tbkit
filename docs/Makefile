# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html-noplot publish Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Builds the pages without running the gallery examples, to check text,
# cross-references and toctrees in seconds.
html-noplot:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -D plot_gallery=0 $(O)

# Builds the full docs (every gallery example, paper listing and
# migration-guide port, each asserting its results) and the coverage report
# at /coverage/, then pushes them to the gh-pages branch. Run it from a clean
# checkout of the commit to publish; NOPUSH=1 commits but does not push.
publish:
	cd .. && for f in paper/listings/*.py docs/source/migration/*.py; do python "$$f" || exit 1; done
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cd .. && python -m pytest tests/ --cov=tbkit --cov-report=html:docs/$(BUILDDIR)/html/coverage
	@# coverage.py writes a .gitignore of "*" that would keep the report out of the commit.
	rm -f $(BUILDDIR)/html/coverage/.gitignore
	touch $(BUILDDIR)/html/.nojekyll
	set -e; \
	git fetch origin gh-pages; \
	tree=$$(mktemp -d); \
	trap 'git worktree remove --force "$$tree"' EXIT; \
	git worktree add --detach "$$tree" origin/gh-pages; \
	rsync -a --delete --exclude .git $(BUILDDIR)/html/ "$$tree"/; \
	git -C "$$tree" add -A; \
	if git -C "$$tree" diff --cached --quiet; then echo "gh-pages is already up to date"; exit 0; fi; \
	git -C "$$tree" commit -q -m "Rebuild docs for $$(git describe --tags --always --dirty)"; \
	if [ -z "$(NOPUSH)" ]; then git -C "$$tree" push origin HEAD:gh-pages; \
	else git -C "$$tree" log -1 --format='%h %s (not pushed)'; fi

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
