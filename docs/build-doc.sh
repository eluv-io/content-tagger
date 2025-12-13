#!/bin/bash

# Builds the documentation for the fabric node's REST API.
#
# This script retrieves all the necessary tools to build the documentation from
# github to a temporary 'build' directory, and patches generation templates and
# stylesheets with our own customizations in doc/api/customizations.
#

set -Eeuo pipefail

command=${0:-}

function showHelp() {
    echo "Usage: $command [OPTIONS]"
    echo
    echo "  -h, --help      show help"
    echo "  -c, --clean     clean build: forces fresh download of build tools"
    echo "  -s, --show-html show resulting HTML in browser"
    echo

    exit $1
}

if ! hash npm >/dev/null 2>&1; then
    echo "Please install Node.js and npm. E.g. 'brew install node'"
    exit 1
fi

show_html=
clean_build=

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    -c | --clean)
        clean_build="true"
        ;;
    -s | --show-html)
        show_html="true"
        ;;
    -h | --help)
        showHelp 0
        ;;
    *) # unknown option
        echo "Unknown option: $key"
        echo
        showHelp 1
        ;;
    esac
    shift
done

proj_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$proj_dir/_build/api"
doc_dir="$proj_dir/api"

[[ ${clean_build} ]] && rm -rf ${build_dir}

if [[ ! -d "$build_dir" ]]; then
    echo "downloading build tools..."
    mkdir -p "$build_dir"
    cd "$build_dir"

    git clone -b add-templates-option https://github.com/lukseven/api2html.git
    cd api2html
    npm install

    echo "done"
fi

cd "$build_dir/api2html"
echo "building api doc..."

# copy original widdershins templates...
rm -rf "$build_dir/widdershins_templates"
cp -rf "$build_dir/api2html/node_modules/widdershins/templates/openapi3" "$build_dir/widdershins_templates"
# ... and patch with our own
cp -f "$doc_dir/customizations/widdershins/templates/"* "$build_dir/widdershins_templates/."

# patch shins with our customizations
cp -rf "$doc_dir/customizations/shins/"* "$build_dir/api2html/node_modules/shins/pub/."

node bin/api2html --summary --customLogo "$doc_dir/logo.png" --templates "$build_dir/widdershins_templates" --customCss -o "$doc_dir/openapi.html" "$doc_dir/openapi.yaml"

# no need to separately bundle api doc with qfab daemon anymore - it gets picks it up through "$doc_dir/embed.go" directly
echo "done"

if [[ ${show_html} ]]; then
    open "$doc_dir/openapi.html"
fi