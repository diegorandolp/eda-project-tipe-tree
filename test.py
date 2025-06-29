import functions_f
print(functions_f.reverse_string("Hola mundo")) 


print(dir(functions_f)) # Deberia de imprimir al funcion reverse y el arkade 

modelo = functions_f.ArkadeModel(
    "gowalla_loc.txt",
    "euclidean",
    0.5,
    10,
    1000,
    100,
    "KnnResults.txt"
)

