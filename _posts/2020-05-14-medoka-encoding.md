---
layout: post
title: 'Using Medoka encoding to compress sparse bitmaps'
tags: [article, compression, bitmap]
featured_image_thumbnail:
featured_image: assets/images/posts/2020/e_l_The_walls_of_Durres.jpg
featured: true
hidden: true
---

In this short article, I am going introduce the idea of Medoka encoding, a simple lossless compression technique which I invented a couple of years ago. In contrast to the [LZ family](https://en.wikipedia.org/wiki/LZ77_and_LZ78) of lossless compression algorithms, Medoka encoding can only be applied to bit arrays, aka bitmaps. The technique is unbelievably simple to code, it is highly parallelizable, very quick in both encoding and decoding, and it usually performed better in my experiments than LZ techniques (of course, more rigorous experiments have to be done for that claim).

Before explaining how the encoding scheme works, let's look at some simple use cases for bitmap compression:
* [Bitmap index](https://en.wikipedia.org/wiki/Bitmap_index#Compression) compression. Bitmap indices are just bit arrays used in databases to answer queries by performing bitwise logical operations. In some cases however, our data columns can be quite sparse. In other words our bitmap would have a lot of zeros, and just a few ones. When dealing with such cases, we usually try to compress the bitmap, so as to not waste unnecessarily space.
* [Bloom Filters](https://en.wikipedia.org/wiki/Bloom_filter). Bloom filters are probabilistic data structures used to test whether an element is a member of a set. They are probabilistic, because false positives can occur while checking for the presence of elements. Bloom filters are also just bit arrays. The ratio of ones and zeros in the bloom filter depends on the chosen parameters of the bloom filter, as well as on the number of elements inserted to the bloom filter. If the difference between zeros and ones is large, it's better to compress the bloom filter, before sending it to another node.

It's important to note that in both mentioned cases, if we were to use a compression technique such as LZ77, we would have first to decode the data, before being able to query it. Medoka encoding does not have this problem, one  can directly participate in bitwise operations without decompressing the data.

Now let's look at how Medoka encoding works, by following simple Python code:

First let's create a random sparse list with ones and zeros:

    import numpy as np
    a = (np.random.rand(100) < 0.1).astype(int).tolist()
    print(a)

The output will look something like this:

    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

To perform Medoka encoding, we first need to transform the bit array. For the sake of it, let's call the transformation "sum_transform":

    def sum_transform(a):
      mapp = {1:1, 0:-1}
      s = []
      n = 1
      prev = a[0]
      for i in a[1:]:
          if i == prev:
              n += 1
          else:
              s.append(n * mapp[prev])
              n = 1
              prev = i
      s.append(n * mapp[prev])
      return s

    b = sum_transform(a)
    print(b)

The output:

    [-7, 2, -3, 1, -3, 1, -19, 1, -7, 2, -2, 1, -18, 1, -20, 1, -11]

As we can see, "sum_transform" takes as input the bit list. To better understand what we're actually doing at this step, look at both the bit list output, as well as the sum_transform output. The function sums up all the zeros that are next to each other and assigns a negative sign to that integer. It does the same thing to the ones, and assigns a positive sign to the integer.

Now comes the final step of the Medoka encoding algorithm:

    def medoka_encoding(a, symbol):
        b = [a[0]]
        for elm in a[1:-1]:
            if elm != symbol:
                b.append(elm)
        b.append(a[-1])
        return b

    c = medoka_encoding(b, 1)
    print(c)

The output:

    [-7, 2, -3, -3, -19, -7, 2, -2, -18, -20, -11]

For the final step of the Medoka encoding algorithm, we simply specify the most frequent symbol from out "sum_transform step", which was the number one. We then remove all ones from sum_transform list. What we get at the end is our final array, that is much more compressable now. We can then compress the array either by using [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding), [Exponential-Golomb coding](https://en.wikipedia.org/wiki/Exponential-Golomb_coding), or any other universal code.

Of course it wouldn't be much of an encoding algorithm if all the steps weren't reversible! To get back to the initial bit array, we simply iterate over the Medoka encoded array. In our case, if two negative numbers occur one after another, we know that we have to insert the number one there. In other words, the sign periodicity of the "sum_transform" step, allows us to infer where a symbol was deleted.

The code for decoding:

    def medoka_decoding(a,symbol):
      b = []
      for i in range(len(a)-1):
          b.append(a[i])
          if np.sign(a[i]) == np.sign(a[i+1]):
              b.append(symbol)
      b.append(a[-1])
      return b

    d = medoka_decoding(c, 1)
    print(d)

Output:

    [-7, 2, -3, 1, -3, 1, -19, 1, -7, 2, -2, 1, -18, 1, -20, 1, -11]

From here we can easily recreate our initial bit array.

In this simple example, I used only a single symbol for the deletion step of the Medoka encoding scheme. We can technically use however two symbols, one symbol with a positive sign, and another with a negative sign. If two negative numbers repeat, we insert the positive symbol during the decoding step. If two positive numbers repeat, we insert the negative symbol during the decoding step. Notice however that if both symbols occur next to each other, only one of them can get deleted, otherwise the computational steps become irreversible.
