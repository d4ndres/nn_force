import h5py

def explorar_h5(archivo):
    with h5py.File(archivo, 'r') as f:
        print(f"\nContenido del archivo: {archivo}")
        print("Estructura del archivo HDF5:\n")

        def explorar_grupo(grupo, nivel=0):
            for clave in grupo:
                objeto = grupo[clave]
                sangria = "  " * nivel
                if isinstance(objeto, h5py.Group):
                    print(f"{sangria}Grupo: {clave}")
                    explorar_grupo(objeto, nivel + 1)
                elif isinstance(objeto, h5py.Dataset):
                    print(f"{sangria}Dataset: {clave}")
                    print(f"{sangria}  Forma: {objeto.shape}")
                    print(f"{sangria}  Tipo de datos: {objeto.dtype}")
        
        explorar_grupo(f)

        print("\nAtributos del archivo:")
        for attr in f.attrs:
            print(f"  {attr}: {f.attrs[attr]}")

# Reemplaza 'tu_archivo.h5' con la ruta a tu archivo
explorar_h5('.\\nn\\pesos_red_neuronal_84_44.h5')
