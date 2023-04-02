function render_table(output, table) {
  if (table.length == 0) {
    output.innerHTML = "<i>empty table</i>"
    return;
  }

  const output_table = document.createElement('table');
  const heading_row = document.createElement('tr');

  const row0 = table[0];
  const table_headings = []
  for (let key of row0.keys()){
    const heading = document.createElement('th');
    heading.innerText = key;
    heading_row.appendChild(heading);
    table_headings.push(key);
  }
  output_table.appendChild(heading_row);

  for (let row of table) {
    const content_row = document.createElement('tr');
    for (let key of table_headings) {
      const element = document.createElement('td');
      const data = row.get(key);
      console.log(key, data);
      element.innerText = JSON.stringify(data);
      content_row.appendChild(element);
    }
    output_table.appendChild(content_row);
  }

  output.appendChild(output_table);
}

async function eval_cell(input, output) {
  console.log(input.value);
  await pyodide.runPython(`
    exe.exec(${JSON.stringify(input.value)})
    result = exe.table.to_dict('records')
  `);
  const output_table = pyodide.globals.get('result').toJs()
  render_table(output, output_table)
}

function create_cell(container) {
  const input = document.createElement("textarea");
  const output = document.createElement("div");
  const cell = document.createElement("div");
  input.addEventListener('keydown', async (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
      await eval_cell(input, output);
      create_cell(container);
    }
  });
  cell.id = "cell" + container.childElementCount;
  cell.appendChild(input);
  cell.appendChild(document.createElement("br"));
  cell.appendChild(output);
  cell.appendChild(document.createElement("br"));
  cell.appendChild(document.createElement("br"));
  container.appendChild(cell);
  input.focus();
}

function setup_notebook() {
  notebook = document.getElementById("notebook");
  console.log(notebook)
  create_cell(notebook, 0);
}

async function main(){
  const progress = document.getElementById("progress");
  progress.innerHTML += "Initializing pyodide<br>"

  let pyodide = await loadPyodide();
  progress.innerHTML += "Installing micropip<br>"
  await pyodide.loadPackage('micropip');
  progress.innerHTML += "Installed micropip<br>"

  await pyodide.runPython(`
    import js
    preload = js.document.getElementById("preload")
    loaded_env = js.document.getElementById("loaded")
    progress = js.document.getElementById("progress")
    exe = None
    result = None
    import micropip
    async def install(progress, pkg):
      progress.innerHTML += f"Installing {pkg}<br>"
      await micropip.install(pkg)
      progress.innerHTML += f"Installed {pkg}<br>"

    async def main():
      global exe
      progress.innerHTML += "Initializing python environment<br>"
      await install(progress, "networkx")
      await install(progress, "pandas")
      await install(progress, "antlr4-python3-runtime")
      progress.innerHTML += "Installing pypher<br>"
      await micropip.install("./dist/pypher_aneeshdurg-0.0.1-py3-none-any.whl", deps=True)
      progress.innerHTML += "Installed pypher<br>"
      progress.innerHTML += "READY!<br>"
      import pypher
      from pypher.pypher import CypherExecutor
      exe = CypherExecutor()

      preload.style = "display:none;"
      loaded_env.style.display = ""
    main()
  `);

  window.pyodide = pyodide;

  setup_notebook();
}
