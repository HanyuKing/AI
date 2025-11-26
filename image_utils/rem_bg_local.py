import rembg

#
# input_path = 'input/cup.jpg'
# output_path = 'output/cup_rembg.jpg'
#
# with open(input_path, 'rb') as i:
#     with open(output_path, 'wb') as o:
#         input = i.read()
#         output = remove(input, force_return_bytes=True)
#         o.write(output)


input_path = 'input/cup.jpg'
output_path = 'output/cup_rembg.jpg'
input = open(input_path, "rb").read()
session = rembg.new_session("birefnet-portrait",**{"model_path": "/Users/rogerswang/.u2net/birefnet-portrait.onnx"})
output = rembg.remove(input, session=session)
with open(output_path, "wb") as o:
    o.write(output)