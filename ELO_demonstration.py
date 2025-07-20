import random
from functools import cmp_to_key

# l = tamanho da array de objetos
# c = tamanho do vetor de características de cada objeto
# r = fator de aleatoriedade nos valores que cada componente do vetor pode assumir
def createObjectsArray(l, c, r):
	objects = []
	
	for i in range(l):
		characteristics = []
		for j in range(c):
			characteristics.append(random.randint(0, r))
		objects.append(characteristics)

	return objects

# Compara vetores a e b de acordo com a ordem lexicográfica extendida
def ELOcompare(a, b):
	sumA = sum(a)
	sumB = sum(b)

	if sumA != sumB:
		return sumA - sumB
	else:
		greatestDiff = 0
		for i in range(len(a)):
			diff = a[i] - b[i]
			if abs(diff) > abs(greatestDiff):
				greatestDiff = diff

		return greatestDiff

# Ordena os vetores de característica em ordem lexicográfica extendida
def ELO(objects):
	return sorted(objects, key=cmp_to_key(ELOcompare))

# Exibe a array de objetos de forma legível
def printObjects(objects):
	for i in range(len(objects)):
		for j in range(len(objects[i])):
			print(objects[i][j], end=" ")
		print(f"-- Soma: {sum(objects[i])}")

def main():
	objects = createObjectsArray(1000, 10, 99)
	print("Antes da ordenação ELO:")
	printObjects(objects)

	# Ordenando com ELO
	objects = ELO(objects)
	
	print("Após a ordenação ELO:")
	printObjects(objects)

main()