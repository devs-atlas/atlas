import Image from 'next/image'
import Layout from '~/components/layout'

import Link from 'next/link'
import { Separator } from '~/components/separator'
import { garamond } from '~/styles/fonts'
import styles from './About.module.css'

export default function About() {
  return (
    <Layout>
      <div className={`${styles.aboutContainer} ${garamond.className}`}>
        <div className={styles.profile}>
          <Image
            src="/nic.webp"
            alt="Nicholas Hoffs profile"
            width={200}
            height={200}
          />
          <div className={`${styles.profileTextContainer}`}>
            <div className={`${styles.name}`}>Nicholas Hoffs</div>
            <div className={`${styles.profileText}`}>
              Nicholas is a second-year student in the University of Virginia’s
              Engineering School. He’s interested in mechanistic
              interpretability, generative sequence models (such as GPT), and
              the “theory of deep learning”.
            </div>
          </div>
        </div>
        <Separator numCircles={7} width={'60%'} />
        <div className={styles.profile}>
          <Image
            src="/barrett.webp"
            alt="Barrett Ruth profile"
            width={200}
            height={200}
          />
          <div className={`${styles.profileTextContainer}`}>
            <div className={`${styles.name}`}>Barrett Ruth</div>
            <div className={`${styles.profileText}`}>
              Barrett is a second-year studying Computer Science and Economics
              at the University of Virginia. He is passionate about writing
              efficient, clean software and the open source community. View his
              website{' '}
              <Link
                style={{ textDecoration: 'underline' }}
                href="https://barrett-ruth.github.io"
              >
                here
              </Link>
              .
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}

// make server side
export async function getServerSideProps() {
  return { props: {} }
}
