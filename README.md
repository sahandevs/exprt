<center>
<p align="center">

<img src="./logo.png" width="300">

</p>
</center>
<p align="center">An expert in expressions!</p>
<h2 align="center">⚠ Work in progress ⚠</h2>


### Design Goals


- plays well with async environments
- fast with the default configuration
- highly tunable for different use-cases
- supports [wirefilter](https://github.com/cloudflare/wirefilter) syntax which itself is based on [wireshark](https://www.wireshark.org/) filter syntax.
- execute other expressions when doing async work
- **easily** embeddable in other languages (current target: go, PHP, typescript, python)
- everything is idempotent and side-effect free by default
- if a user writes an unoptimized code, compile and optimize it to the optimized version. if the compiler knows a code is not optimized, it should just compile it to the optimized version instead of showing a message to user.
- all values are inferred. no dynamic functions or values


#### Maybe?

- PGO or JIT
- string interning?
- directly compile to wasm format (instead of using our own vm)
- usable as library and cli (something like awk?)