function render_table(output, table) {
  if (table.length == 0) {
    output.innerHTML = "<i>empty table</i>"
    return;
  }

  const output_table = document.createElement('table');
  const heading_row = document.createElement('tr');

  const row0 = table[0];
  const table_headings = []
  for (let key of Object.getOwnPropertyNames(row0)){
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
      const data = row[key];
      element.innerText = JSON.stringify(data);
      content_row.appendChild(element);
    }
    output_table.appendChild(content_row);
  }

  output.appendChild(output_table);
}

async function eval_cell(input, output) {
  output.innerHTML = "";

  window.stderr = []
  try {
    await pyodide.runPython(`
      try:
        exe.exec(${JSON.stringify(input.value)})
        result = exe.table_to_json()
      except Exception as e:
        print(e, file=sys.stderr)
        raise Exception from e
    `);
    const output_table = JSON.parse(pyodide.globals.get('result'))
    render_table(output, output_table)
  } catch (e){
    const summary = document.createElement('summary');
    if (window.stderr.length) {
      for (let msg of window.stderr) {
        const error_msg = document.createElement('p');
        error_msg.className = "errormsg";
        error_msg.innerText = msg;
        summary.appendChild(error_msg);
      }
    } else {
      const error_msg = document.createElement('p');
      error_msg.className = "errormsg";
      error_msg.innerText = "UNKNOWN EXECUTION ERROR";
      summary.appendChild(error_msg);
    }

    const details = document.createElement('details');
    const errors = document.createElement('code');
    errors.innerText = e;
    details.appendChild(errors);

    output.appendChild(summary);
    output.appendChild(details);
  }
}

function create_cell(container) {
  const input = document.createElement("textarea");
  input.className = "cellinput";
  const run = document.createElement("button");
  run.innerText = "run"
  const output = document.createElement("div");
  const cell = document.createElement("div");
  const count = container.childElementCount;
  cell.id = "cell" + container.childElementCount;

  cell.addEventListener('focus', () => {
    input.focus();
  });

  const evaluate = async () => {
    await eval_cell(input, output);
    const nextid = count + 1;
    const next_el = document.getElementById("cell" + nextid);
    if (next_el) {
      next_el.focus();
    } else {
      create_cell(container);
    }
  };
  input.addEventListener('keydown', async (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
      evaluate();
    }
  });
  run.addEventListener('click', evaluate);

  cell.appendChild(input);
  cell.appendChild(document.createElement("br"));
  cell.appendChild(run);
  cell.appendChild(document.createElement("br"));
  cell.appendChild(document.createElement("br"));
  cell.appendChild(output);
  cell.appendChild(document.createElement("br"));
  cell.appendChild(document.createElement("br"));
  container.appendChild(cell);
  input.focus();
  container.scrollTop = container.scrollHeight;
}

function setup_notebook() {
  notebook = document.getElementById("notebook");
  create_cell(notebook, 0);
}

async function main(){
  const progress = document.getElementById("progress");
  progress.innerHTML += "Initializing pyodide<br>"

  let pyodide = await loadPyodide();
  progress.innerHTML += "Installing micropip<br>"
  await pyodide.loadPackage('micropip');
  progress.innerHTML += "Installed micropip<br>"

  window.stderr = [];
  console.warn = (x) => {
    window.stderr.push(x);
  };

  const originalConsoleLog = console.log;
  console.log = (s) => {
    progress.innerText += "<pyodide>: " + s + "\n";
  };

  await pyodide.runPython(`
    import sys

    import js
    preload = js.document.getElementById("preload")
    loaded_env = js.document.getElementById("loaded")
    progress = js.document.getElementById("progress")
    exe = None
    result = None
    import micropip
    async def main():
      global exe
      progress.innerHTML += "Initializing python environment<br>"
      progress.innerHTML += "Installing spycy (may take a few minutes)<br>"
      await micropip.install("./dist/spycy_aneeshdurg-0.0.2-py3-none-any.whl", deps=True)
      progress.innerHTML += "Installed spycy<br>"
      progress.innerHTML += "READY!<br>"
      import spycy
      from spycy.spycy import CypherExecutor
      exe = CypherExecutor()

      preload.style = "display:none;"
      loaded_env.style.display = ""
    main()
  `);

  console.log = originalConsoleLog;

  window.pyodide = pyodide;

  setup_notebook();
}
