/*
 * Prilagojeno s
 * https://github.com/awreece/pdfcrack-opencl/blob/master/md5.cl
 * https://github.com/Rich5/oclHashcat/tree/master/OpenCL
 *
 * Za razumevanje:
 * https://www.ietf.org/rfc/rfc1321.txt
 * https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_C.html
 * https://en.wikipedia.org/wiki/MD5
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// This will be set in the Rust executable based on CLI arguments!
#define CALCULATE_HASHRATE _hashrate_bool_def

/* The basic MD5 functions */
#define F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

/* The MD5 transformation for all four rounds. */
#define STEP(f, a, b, c, d, x, t, s)                                                                                   \
	(a) += f((b), (c), (d)) + (x) + (t);                                                                               \
	(a) = (((a) << (s)) | (((a)&0xffffffff) >> (32 - (s))));                                                           \
	(a) += (b);

#define GET(i) (key[(i)])

static void md5_round(uint *internal_state, const uint *key) {
	uint a, b, c, d;
	a = internal_state[0];
	b = internal_state[1];
	c = internal_state[2];
	d = internal_state[3];

	/* Round 1 */
	STEP(F, a, b, c, d, GET(0), 0xd76aa478, 7)
	STEP(F, d, a, b, c, GET(1), 0xe8c7b756, 12)
	STEP(F, c, d, a, b, GET(2), 0x242070db, 17)
	STEP(F, b, c, d, a, GET(3), 0xc1bdceee, 22)
	STEP(F, a, b, c, d, GET(4), 0xf57c0faf, 7)
	STEP(F, d, a, b, c, GET(5), 0x4787c62a, 12)
	STEP(F, c, d, a, b, GET(6), 0xa8304613, 17)
	STEP(F, b, c, d, a, GET(7), 0xfd469501, 22)
	STEP(F, a, b, c, d, GET(8), 0x698098d8, 7)
	STEP(F, d, a, b, c, GET(9), 0x8b44f7af, 12)
	STEP(F, c, d, a, b, GET(10), 0xffff5bb1, 17)
	STEP(F, b, c, d, a, GET(11), 0x895cd7be, 22)
	STEP(F, a, b, c, d, GET(12), 0x6b901122, 7)
	STEP(F, d, a, b, c, GET(13), 0xfd987193, 12)
	STEP(F, c, d, a, b, GET(14), 0xa679438e, 17)
	STEP(F, b, c, d, a, GET(15), 0x49b40821, 22)

	/* Round 2 */
	STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
	STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
	STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
	STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
	STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
	STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
	STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
	STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
	STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
	STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
	STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
	STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
	STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
	STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
	STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
	STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

	/* Round 3 */
	STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
	STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
	STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
	STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
	STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
	STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
	STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
	STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
	STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
	STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
	STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
	STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
	STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
	STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
	STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
	STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)

	/* Round 4 */
	STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
	STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
	STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
	STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
	STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
	STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
	STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
	STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
	STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
	STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
	STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
	STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
	STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
	STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
	STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
	STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

	internal_state[0] += a;
	internal_state[1] += b;
	internal_state[2] += c;
	internal_state[3] += d;
}

void md5(const char *restrict msg, uint length_bytes, uint *restrict out) {
	uint i, bytes_left;
	char key[64];

	out[0] = 0x67452301;
	out[1] = 0xefcdab89;
	out[2] = 0x98badcfe;
	out[3] = 0x10325476;

	for (bytes_left = length_bytes; bytes_left >= 64; bytes_left -= 64, msg = &msg[64])
		md5_round(out, (const uint *)msg);

	for (i = 0; i < bytes_left; i++) key[i] = msg[i];

	key[bytes_left++] = 0x80;

	if (bytes_left <= 56) {
		for (i = bytes_left; i < 56; key[i++] = 0) {}
	} else {
		for (i = bytes_left; i < 64; key[i++] = 0) {}
		md5_round(out, (uint *)key);
		for (i = 0; i < 56; key[i++] = 0) {}
	}

	ulong *len_ptr = (ulong *)&key[56];

	*len_ptr = length_bytes * 8;

	md5_round(out, (uint *)key);
}

__kernel void hash(__private ulong const  problem_space,
				   __private uint const	  number_of_zeroes,
				   __private uint const	  word_length,
				   __global char *const	  word,
				   __private ushort const nonce_len,
				   __global int			*work_flag,
				   __global uint		 *solution,
				   __global ulong		  *nonce_solution
#if CALCULATE_HASHRATE
				   ,
				   __global ulong *total_hashes,
				   __local ulong	 *partial_hash_sum
#endif
) {

	__private uint	i;
	__private ulong g_id, nonce_word;

	/** Variable array lengths are not supported in OpenCL */
	__private char in[64], out[16];

	g_id = get_global_id(0);

#if CALCULATE_HASHRATE
	__private ulong hash_count = 0;
#endif

	while (g_id < problem_space && !atomic_max(work_flag, 0)) {
		/** Construct the message we wish to hash */
		nonce_word = g_id;
		for (i = 0; i < word_length; i++) in[i] = word[i];
		// shifting to right 8 places = division by 256, and 255 = modulo by 256
		for (i = 0; i < nonce_len; i++, nonce_word = nonce_word >> 8) in[i + word_length] = (char)(nonce_word & 255);

		/* Store the MD5 Hash in the out buffer, remember its length is 16 uchars or 4 uints */
		md5(in, (uint)(word_length + nonce_len), (uint *)out);

		/** Count the number of zeroes at the end */
		i = 15;

		// The while loop counts only HEX which equls 0x00
		while (i >= 1 && out[i--] == 0) {}

		// !(out[i + 1] & 15) counts HEX characters like 0xA0, 0xB1 etc. but instead of using modulo operator it uses a
		// bitmask for performance reasons
		if (28 - 2 * i + !(out[i + 1] & 15) == number_of_zeroes) {
			if (!atomic_cmpxchg(work_flag, 0, 1)) {
				for (i = 0; i < 4; i++) solution[i] = ((uint *)out)[i];
				*nonce_solution = g_id;
			} else
				break;
		}

		/** Increment the global id, because the problem space is probably bigger than the number of threads */
		g_id += get_global_size(0);

#if CALCULATE_HASHRATE
		hash_count++;
#endif
	}

#if CALCULATE_HASHRATE
	partial_hash_sum[get_local_id(0)] = hash_count;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0) {
		ulong sum = 0;
		for (int i = 0; i < get_local_size(0); i++) sum += partial_hash_sum[i];
		atomic_add(total_hashes, sum);
	}
#endif
}
