# Aplikacja BFS z CUDA

## Opis
Aplikacja wykonuje algorytm przeszukiwania grafu wszerz (BFS) z wykorzystaniem CUDA. Przyjmuje graf w postaci listy krawędzi z pliku wejściowego, a wyniki zapisuje do określonego pliku wyjściowego. Możliwe jest określenie wierzchołka startowego i końcowego dla algorytmu.

## Użycie
Aplikacja wymaga podania czterech argumentów wiersza poleceń: nazwy pliku wejściowego, nazwy pliku wyjściowego, wierzchołka startowego i końcowego. Możliwe jest także wyświetlenie pomocy przy użyciu argumentu `-help`.

### Składnia
[nazwa_programu] -f [nazwa_pliku_grafu] -o [nazwa_pliku_wyjściowego] -s [wierzchołek_startowy] -e [wierzchołek_końcowy]

### Przykład
./bfs -f graf.txt -o wynik.txt -s 0 -e 10

W powyższym przykładzie, program wczyta graf z pliku `graf.txt`, przeprowadzi algorytm BFS od wierzchołka 0 do 10, a wynik zapisze do pliku `wynik.txt`.

## Pomoc
Aby wyświetlić pomoc dotyczącą używania programu, użyj:
[nazwa_programu] -help


## Autor
Bartosz Maj