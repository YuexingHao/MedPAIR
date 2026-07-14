#!/bin/bash
# Script to compile Appendix.tex to PDF
# Usage: bash compile_appendix.sh

cd "$(dirname "$0")"

echo "Compiling Appendix.tex to PDF..."

# Try different LaTeX engines in order of preference
for engine in xelatex pdflatex lualatex; do
    if command -v $engine &> /dev/null; then
        echo "Using $engine..."
        $engine -shell-escape -interaction=nonstopmode Appendix.tex

        if [ -f "Appendix.pdf" ]; then
            echo "✓ Successfully compiled Appendix.pdf"
            exit 0
        fi
    fi
done

echo "✗ Failed to compile. No suitable LaTeX engine found."
echo ""
echo "Please install TeX Live or MacTeX and try again:"
echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
echo "  macOS: brew install mactex"
echo "  Fedora/CentOS: sudo dnf install texlive-scheme-full"
exit 1
