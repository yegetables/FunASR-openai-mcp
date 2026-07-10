Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "C:\Users\yegetables\scoop\apps\python312\current\python.exe C:\Users\yegetables\work\funasr-repo\examples\openai_api\server.py --device cuda --port 8221 --host 0.0.0.0", 0, False
