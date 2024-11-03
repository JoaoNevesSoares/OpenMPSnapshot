/* Copyright (c) 2023 Adriano Angelone
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * This file is part of das-c.
 *
 * This file may be used under the terms of the GNU General Public License
 * version 3.0 as published by the Free Software Foundation and appearing in
 * the file LICENSE included in the packaging of this file. Please review the
 * following information to ensure the GNU General Public License version 3.0
 * requirements will be met:
 * http://www.gnu.org/copyleft/gpl.html.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE. */

#include "das-c/common.h"
#include "das-c/parse.h"
#include <omp.h>

// Parses the `idx_thread`-th chunk of a file, filling the corresponding rows
// in `tab` based on the information in `info`.
//
// Returns 0 on success, the line number where the error occurred (indexed from
// 1) on failure.
size_t
parse_chunk_openmp(table *tab, const parse_info *info, const size_t idx_thread)
{
  size_t retval = 0;

  FILE *file = fopen(info->filename, "r");
  fseek(file, info->pos[idx_thread], 0);

  // Calculation of starting index
  size_t start = 0;
  for (size_t it = 0; it < idx_thread; ++it)
    start += info->chunks[it];

  char line[DASC_MAX_LINE_LENGTH];

  size_t ir = 0;
  do
  {
    const size_t row = start + ir;

    fgets(line, DASC_MAX_LINE_LENGTH, file);
    if (is_comment(line))
      continue;

    if (parse_line(tab->data[row], line, info->msk) == EXIT_FAILURE)
    {
      retval = (row + 1);
      break;
    }

    ++ir;
  } while (ir < info->chunks[idx_thread]);

  fclose(file);
  return retval;
}

size_t parse_openmp(table *tab, const parse_info *info)
{
  init_table(tab, info->rows, info->msk->n_active);

  size_t retval = 0;
  size_t *res = malloc(info->n_threads * sizeof(size_t));

  omp_set_dynamic(0);
  omp_set_num_threads((int)(info->n_threads));

#pragma omp parallel for
  for (size_t it = 0; it < info->n_threads; ++it)
    res[it] = parse_chunk_openmp(tab, info, it);

  for (size_t it = 0; it < info->n_threads; ++it)
  {
    if (res[it])
    {
      retval = res[it];
      break;
    }
  }

  free(res);
  return retval;
}
