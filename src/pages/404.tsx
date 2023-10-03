import Layout from '~/components/layout'
import VertSeparator from '~/components/vertseparator'
import { fragment } from '~/styles/fonts'
import styles from './_404.module.css'

export default function _404() {
  return (
    <Layout>
      <div className={`${fragment.className} ${styles.notFoundContainer}`}>
        <span
          className={`${styles.notFound}`}
          onClick={() => window.history.back()}
          style={{ cursor: 'pointer' }}
        >
          <span>404</span>
          <VertSeparator />
          <span>Not Found</span>
        </span>
      </div>
    </Layout>
  )
}
